from __future__ import annotations

import ast
import hashlib
import io
import json
import multiprocessing
import random
from operator import attrgetter, itemgetter
from pathlib import Path
from queue import Empty

import pandas
import pyarrow.parquet as pq
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from . import registry
from .registry import (
    AudioProbeResult,
    DataClipScan,
    DataManifest,
    DataWalk,
    LatentCrop,
    LatentScan,
    SampleClip,
    SampleLatent,
    SpecTrainData,
    VaeBatch,
    get_cache_clip_paths,
    get_cache_header_path,
    get_cache_shard_path,
    get_domain,
)


class AudioProbe:
    # Header probe through torchcodec, where torchaudio lacks info().
    def get_codec_probe(self, source):
        # images without torchaudio.info are the only torchcodec users
        from torchcodec.decoders import AudioDecoder

        decoder = AudioDecoder(source)
        meta = decoder.metadata
        # header duration; mp3 frame counts land within half a percent
        num_frames = round(meta.duration_seconds * meta.sample_rate)
        probe = AudioProbeResult(sample_rate=meta.sample_rate, num_frames=num_frames)
        return probe

    # Sample rate and frame count; torchaudio 2.9 dropped info().
    def get_audio_probe(self, source):
        info_fn = getattr(torchaudio, "info", None)
        if info_fn is not None:
            meta = info_fn(source)
            probe = AudioProbeResult(sample_rate=meta.sample_rate, num_frames=meta.num_frames)
        else:
            probe = self.get_codec_probe(source)
        return probe


class ShardTable:
    # Open one pinned shard, digest checked by the hub.
    def open_shard_table(self, spec: registry.DataShards, shard: int, cache: Path):
        file_name = f"data/{spec.split}-{shard:05d}-of-{spec.shards:05d}.parquet"
        cache_dir = str(cache)
        path = hf_hub_download(
            spec.repo,
            file_name,
            repo_type="dataset",
            revision=spec.revision,
            cache_dir=cache_dir,
        )
        table = pq.ParquetFile(path)
        return table


# Audio-blob reader over one parquet shard, rewindable.
class ShardCursor:
    def __init__(self, dataset: str, shard: int, root: Path):
        self.spec = registry.get_version(dataset)
        self.shard = shard
        self.root = root
        self.column = self.spec.audio
        self.shard_table = ShardTable()
        self.open_stream()

    # Restart the row stream at the head of the shard.
    def open_stream(self):
        cache_dir = self.root / "hf_cache"
        table = self.shard_table.open_shard_table(self.spec, self.shard, cache_dir)
        self.batches = table.iter_batches(batch_size=16, columns=[self.column])
        self.row_list = []
        self.start = 0

    # Blob bytes of one row; StopIteration past the shard end.
    def load_blob(self, row: int):
        if row < self.start:
            # a random draw may revisit a row already streamed
            self.open_stream()
        while row >= self.start + len(self.row_list):
            self.start += len(self.row_list)
            record_batch = next(self.batches)
            self.row_list = record_batch.to_pylist()
        row_dict = self.row_list[row - self.start]
        cell = row_dict[self.column]
        blob = None
        if cell:
            blob = cell.get("bytes")
        return blob


# Locator to waveform: folder file, staged wav, or parquet row.
class ShardAudioSet:
    def __init__(self, root: Path):
        self.root = root
        self.cursor_dict = {}
        self.locator = registry.DataLocator()
        self.probe = AudioProbe()
        self.bad_source_list = []

    # Fold channels, then window, crop or pad.
    def apply_clip_shape(
        self, wav: torch.Tensor, channels: int, num_samples: int, start_sample: int = 0,
        fold: str = "stereo",
    ):
        if channels == 1 or fold == "mean":
            wav = wav.mean(dim=0, keepdim=True)  # (1, S)
        if fold == "first":
            wav = wav[:1]                        # (1, S)
        if wav.shape[0] == 1 and channels > 1:
            wav = wav.repeat(channels, 1)        # (C, S)
        if wav.shape[0] > channels:
            wav = wav[:channels]                 # (C, S)
        tail = max(wav.shape[-1] - start_sample, 0)
        valid = min(tail, num_samples)
        out = torch.zeros(channels, num_samples)  # (C, N)
        if valid:
            out[:, :valid] = wav[:, start_sample:start_sample + valid]
        return out, valid

    # Decode one target-rate window, seeking when rates match.
    def load_audio_segment(
        self, source, sample_rate: int, channels: int, start_sample: int, num_samples: int,
        fold: str = "stereo",
    ):
        source_text = str(source)
        wav = None
        native = sample_rate
        try:
            probe = self.probe.get_audio_probe(source_text)
            native = probe.sample_rate
            if native == sample_rate:
                wav, _ = torchaudio.load(
                    source_text, frame_offset=start_sample, num_frames=num_samples
                )
            else:
                # Full decode preserves resampling context and phase.
                wav, _ = torchaudio.load(source_text)
        except (RuntimeError, OSError, ValueError) as error:
            print(f"audio decode failed, zero fill {source}: {error!r}")
            self.bad_source_list.append(source_text)
        if wav is None:
            silence = torch.zeros(channels, num_samples)
            segment = (silence, 0)
        elif native == sample_rate:
            segment = self.apply_clip_shape(wav, channels, num_samples, fold=fold)
        else:
            wav = torchaudio.functional.resample(wav, native, sample_rate)  # (C, S)
            segment = self.apply_clip_shape(wav, channels, num_samples, start_sample, fold)
        return segment

    # Cursor for one shard; one open shard per dataset.
    def open_shard_cursor(self, dataset: str, shard: int):
        key = (dataset, shard)
        if key not in self.cursor_dict:
            kept_dict = {}
            for cursor_key, cursor_item in self.cursor_dict.items():
                if cursor_key[0] != dataset:
                    kept_dict[cursor_key] = cursor_item
            self.cursor_dict = kept_dict
            self.cursor_dict[key] = ShardCursor(dataset, shard, self.root)
        cursor = self.cursor_dict[key]
        return cursor

    # Decode one parquet row window straight from the blob.
    def load_row_segment(
        self, dataset: str, shard: int, row: int, sample_rate: int,
        channels: int, start_sample: int, num_samples: int, fold: str = "stereo",
    ):
        locator = f"{dataset}#{shard}#{row}"
        stop = (start_sample + num_samples) / sample_rate + 0.5
        wav = None
        native = sample_rate
        try:
            # absent shard or row is missing data, not fatal
            cursor = self.open_shard_cursor(dataset, shard)
            blob = cursor.load_blob(row)
            if blob:
                handle = io.BytesIO(blob)
                probe = self.probe.get_audio_probe(handle)
                native = probe.sample_rate
                handle.seek(0)
                # bounded decode, so a long clip costs one window
                num_frames = int(stop * native)
                wav, _ = torchaudio.load(handle, num_frames=num_frames)
            else:
                print(f"audio decode failed, zero fill {locator}: row carries no audio bytes")
        except (RuntimeError, OSError, ValueError, KeyError, StopIteration) as error:
            print(f"audio decode failed, zero fill {locator}: {error!r}")
        if wav is None:
            self.bad_source_list.append(locator)
            silence = torch.zeros(channels, num_samples)
            segment = (silence, 0)
        else:
            if native != sample_rate:
                wav = torchaudio.functional.resample(wav, native, sample_rate)  # (C, S)
            segment = self.apply_clip_shape(wav, channels, num_samples, start_sample, fold)
        return segment

    # Resolve one locator to an audio file on disk.
    def get_source_path(self, source: str):
        parts = self.locator.parse_locator(source)
        if parts.middle == "file":
            spec = registry.get_version(parts.dataset)
            path = spec.get_path(self.root, parts.ident)
        else:
            path = self.root / "wav_cache" / f"{source}.wav"
        return path

    # Decode a window from a folder file or staged wav.
    def load_path_segment(
        self, source: str, sample_rate: int, channels: int,
        start_sample: int, num_samples: int, fold: str,
    ):
        path = None
        try:
            path = self.get_source_path(source)
        except KeyError as error:
            print(f"source missing, zero fill {source}: {error!r}")
            self.bad_source_list.append(source)
        if path is None:
            silence = torch.zeros(channels, num_samples)
            segment = (silence, 0)
        else:
            segment = self.load_audio_segment(
                path, sample_rate, channels, start_sample, num_samples, fold,
            )
        return segment

    # Decode a bounded target-rate window from one source clip.
    def load_segment(
        self, source: str, sample_rate: int, channels: int,
        start_sample: int, num_samples: int, fold: str = "stereo",
    ):
        parts = self.locator.parse_locator(source)
        staged = self.root / "wav_cache" / f"{source}.wav"
        # staged wavs keep the numerics of finished slots
        if parts.middle != "file" and not staged.is_file():
            shard = int(parts.middle)
            row = int(parts.ident)
            segment = self.load_row_segment(
                parts.dataset, shard, row,
                sample_rate, channels, start_sample, num_samples, fold,
            )
        else:
            segment = self.load_path_segment(
                source, sample_rate, channels, start_sample, num_samples, fold,
            )
        return segment


class AudioStreamReader:
    # Decode one request batch and queue it as numpy.
    def load_stream_batch(
        self, audio: ShardAudioSet, request_list: list, sample_rate: int,
        channels: int, num_samples: int, out_queue,
    ):
        wav_list = []
        valid_list = []
        for request_item in request_list:
            source, offset, fold = request_item
            wav, valid = audio.load_segment(
                source, sample_rate, channels, offset, num_samples, fold
            )
            wav_list.append(wav)
            valid_list.append(valid)
        wav_batch = torch.stack(wav_list)
        # numpy payloads copy by value; the reader can exit
        wav_array = wav_batch.numpy()
        out_queue.put((wav_array, valid_list))

    # Reader-process loop: decode batches into the queue.
    def load_stream_batches(
        self, root: str, request_groups: list, sample_rate: int, channels: int,
        num_samples: int, batch_clips: int, out_queue,
    ):
        audio = ShardAudioSet(Path(root))
        for request_group in request_groups:
            # batches never straddle a shard, to match the encode loop
            for start_idx in range(0, len(request_group), batch_clips):
                request_list = request_group[start_idx:start_idx + batch_clips]
                self.load_stream_batch(
                    audio, request_list, sample_rate, channels, num_samples, out_queue,
                )
        out_queue.put(None)


# Start one audio reader process feeding a bounded queue.
def open_batch_stream(
    root: Path, request_groups: list, sample_rate: int, channels: int,
    num_samples: int, batch_clips: int, depth: int = 4,
):
    group_list = []
    for request_group in request_groups:
        group_list.append(list(request_group))
    context = multiprocessing.get_context("spawn")
    stream = context.Queue(maxsize=depth)
    stream_reader = AudioStreamReader()
    reader = context.Process(
        target=stream_reader.load_stream_batches,
        args=(str(root), group_list, sample_rate, channels, num_samples, batch_clips, stream),
        daemon=True,
    )
    reader.start()
    return reader, stream


# Next decoded batch; a dead reader surfaces queue.Empty.
def get_stream_batch(reader, stream):
    batch = None
    waiting = True
    while waiting:
        try:
            batch = stream.get(timeout=30)
            waiting = False
        except Empty:
            if not reader.is_alive():
                # an empty queue behind a dead reader raises Empty here
                batch = stream.get(block=False)
                waiting = False
    result = None
    if batch is not None:
        wav_array, valid_list = batch
        wavs = torch.from_numpy(wav_array)  # (B, C, N)
        result = (wavs, valid_list)
    return result


class PromptBuilder:
    # Metadata types to one string each, empties dropped.
    def collect_prompt_items(self, metadata: tuple, stream: random.Random):
        item_list = []
        for metadata_item in metadata:
            key, values = metadata_item
            pool = []
            for value_item in values:
                text = str(value_item)
                text = text.strip()
                if text:
                    pool.append(text)
            if pool:
                if len(pool) > 1:
                    # the paper shuffles list values such as tags or genres
                    stream.shuffle(pool)
                joined = ", ".join(pool)
                item_list.append((key, joined))
        return item_list

    # Metadata text as written, comma joined.
    def make_verbatim_prompt(self, metadata: tuple):
        text_list = []
        for metadata_item in metadata:
            for value_item in metadata_item[1]:
                text = str(value_item)
                text = text.strip()
                if text:
                    text_list.append(text)
        prompt = ", ".join(text_list)
        return prompt

    # Draw a subset, shuffle it, type it, then case it.
    def make_drawn_prompt(self, dataset: str, item_list: list, stream: random.Random):
        n_keep = stream.randint(1, len(item_list))
        keep_list = stream.sample(item_list, n_keep)
        # the paper shuffles the resulting order too
        stream.shuffle(keep_list)
        spec = registry.get_version(dataset)
        typed = stream.random() < spec.prompt_typed_ratio
        part_list = []
        for keep_item in keep_list:
            key, value = keep_item
            if typed:
                part_list.append(f"{key}: {value}")
            else:
                part_list.append(value)
        text = ", ".join(part_list)
        # lower, upper or unchanged; the paper gives no split
        roll = stream.randrange(3)
        if roll == 0:
            prompt = text.lower()
        elif roll == 1:
            prompt = text.upper()
        else:
            prompt = text
        return prompt

    # Random metadata subset, shuffled and comma joined.
    def make_prompt(self, dataset: str, metadata: tuple, seed: int, policy: str):
        if policy == registry.VERBATIM_POLICY:
            prompt = self.make_verbatim_prompt(metadata)
        else:
            stream = random.Random(seed)
            item_list = self.collect_prompt_items(metadata, stream)
            prompt = ""
            if item_list:
                prompt = self.make_drawn_prompt(dataset, item_list, stream)
        return prompt


# Raw dataset rows from hub shards, folders and listings.
class RawHfDataset:
    def __init__(self, cfg: registry.DataWalkConfig = registry.DataWalkConfig()):
        self.cfg = cfg
        self.sidecars = {}
        self.listings = {}
        self.locator = registry.DataLocator()
        self.probe = AudioProbe()
        self.shard_table = ShardTable()
        self.bad_item_list = []

    # Per-clip noise seed from content hash, world-size independent.
    @staticmethod
    def make_eps_seed(sample_hash: str, seed: int):
        hash_value = int(sample_hash[:16], 16)
        eps_seed = (hash_value ^ seed) & ((1 << 63) - 1)
        return eps_seed

    # A later hash slice, so prompts miss the noise draws.
    @staticmethod
    def make_prompt_seed(sample_hash: str, seed: int):
        hash_value = int(sample_hash[16:32], 16)
        prompt_seed = (hash_value ^ seed) & ((1 << 63) - 1)
        return prompt_seed

    # The trailing slice, so crops miss both other draws.
    @staticmethod
    def make_crop_seed(sample_hash: str, seed: int):
        hash_value = int(sample_hash[32:40], 16)
        crop_seed = (hash_value ^ seed) & ((1 << 63) - 1)
        return crop_seed

    # Random metadata subset, shuffled and comma joined.
    @staticmethod
    def make_prompt(
        dataset: str, metadata: tuple, seed: int,
        policy: str = registry.prompt_policy,
    ):
        builder = PromptBuilder()
        prompt = builder.make_prompt(dataset, metadata, seed, policy)
        return prompt

    # Stripped text of one cell, empty when unset.
    def make_text(self, raw):
        text = str(raw or "")
        text = text.strip()
        return text

    # One stripped value as a tuple; empty text drops out.
    def make_value_tuple(self, raw):
        text = self.make_text(raw)
        value_tuple = ()
        if text:
            value_tuple = (text,)
        return value_tuple

    # Stripped tags as a tuple, empty tags dropped.
    def make_tag_tuple(self, raw):
        tag_list = []
        for tag_item in raw or ():
            text = str(tag_item)
            text = text.strip()
            if text:
                tag_list.append(text)
        tag_tuple = tuple(tag_list)
        return tag_tuple

    # A csv cell holding a python list of genre ids.
    def parse_genre_ids(self, cell: str):
        id_list = []
        try:
            parsed = ast.literal_eval(cell or "[]")
            for value_item in parsed:
                id_list.append(str(value_item))
        except (ValueError, TypeError, SyntaxError, RecursionError) as error:
            print(f"genre cell unparsed: {error!r}")
            self.bad_item_list.append(cell)
            id_list = []
        genre_tuple = tuple(id_list)
        return genre_tuple

    # Genre id to title, so prompts carry words not ids.
    def load_genre_names(self, path: Path):
        genre_df = pandas.read_csv(path, dtype=str, keep_default_na=False)
        genre_dict = {}
        for genre_id, title in zip(genre_df["genre_id"], genre_df["title"]):
            genre_dict[genre_id] = title
        return genre_dict

    # One FMA track row to its five metadata types.
    def make_fma_items(self, row: list, column_dict: dict, genre_dict: dict):
        item_list = []
        for column_item in self.cfg.fma_column_list:
            name, key = column_item
            cell = ""
            if key in column_dict:
                cell = row[column_dict[key]]
            if name == "genres":
                genre_list = []
                for genre_item in self.parse_genre_ids(cell):
                    genre_list.append(genre_dict.get(genre_item, genre_item))
                value_tuple = tuple(genre_list)
            elif name == "year":
                # date_released is a timestamp; the paper wants the year
                stripped = cell.strip()
                year = stripped[:4]
                value_tuple = ()
                if year.isdigit():
                    value_tuple = (year,)
            else:
                value_tuple = self.make_value_tuple(cell)
            item_list.append((name, value_tuple))
        items = tuple(item_list)
        return items

    # Track id to the five metadata types, genres as names.
    def load_fma_metadata(self, root: Path):
        genre_dict = self.load_genre_names(root / "genres.csv")
        # two header rows, then one row naming only the index
        track_df = pandas.read_csv(root / "tracks.csv", header=None, dtype=str, keep_default_na=False)
        value_array = track_df.values
        table = value_array.tolist()
        column_dict = {}
        for column_idx, column_pair in enumerate(zip(table[0], table[1])):
            column_dict[column_pair] = column_idx
        found = {}
        for row_item in table[3:]:
            track_id = ""
            if row_item:
                track_id = row_item[0].strip()
            if track_id:
                found[track_id] = self.make_fma_items(row_item, column_dict, genre_dict)
        return found

    # Freesound record to title, description and tags.
    def make_freesound_items(self, record: dict):
        item_list = []
        for column_item in self.cfg.freesound_column_list:
            name, key = column_item
            value_tuple = self.make_value_tuple(record.get(key))
            item_list.append((name, value_tuple))
        tag_tuple = self.make_tag_tuple(record.get("tags"))
        item_list.append(("tags", tag_tuple))
        items = tuple(item_list)
        return items

    # Parse one jsonl line into the table; one when skipped.
    def parse_freesound_line(self, line: str, found: dict):
        n_skipped = 0
        parsed = False
        record = None
        try:
            record = json.loads(line)
            parsed = True
        except json.JSONDecodeError as error:
            n_skipped = 1
            print(f"freesound metadata line skipped: {error!r}")
            self.bad_item_list.append(line)
        if parsed:
            items = self.make_freesound_items(record)
            record_id = record.get("id")
            found[str(record_id)] = items
        return n_skipped

    # Sound id to title, description and shuffled-later tags.
    def load_freesound_metadata(self, path: Path):
        found = {}
        skipped = 0
        with path.open(encoding="utf-8") as handle:
            for line_item in handle:
                line = line_item.strip()
                if line:
                    skipped += self.parse_freesound_line(line, found)
        if skipped:
            print(f"freesound metadata: {skipped} lines skipped")
        return found

    # Audio stem to caption, title and description.
    def load_captions_metadata(self, path: Path):
        text = path.read_text(encoding="utf-8")
        record = json.loads(text)
        row_list = record
        if isinstance(record, dict):
            row_list = record["data"]
        found = {}
        for row_item in row_list:
            item_list = []
            for name_item in ("caption", "title", "description"):
                value_tuple = self.make_value_tuple(row_item.get(name_item))
                item_list.append((name_item, value_tuple))
            # subset ids carry stray audio suffixes, files do not
            row_id = str(row_item.get("id") or "")
            stem = Path(row_id).stem
            found[stem] = tuple(item_list)
        return found

    # Use the caption text inside Gemini's serialized annotation.
    def parse_ifcaps_caption(self, value):
        text = self.make_text(value)
        caption = text
        if text.startswith("{"):
            try:
                record = json.loads(text)
                if isinstance(record, dict) and "caption" in record:
                    caption = self.make_text(record["caption"])
            except json.JSONDecodeError:
                self.bad_item_list.append(text)
        return caption

    # Unambiguous extracted stems to their caption rows.
    def load_ifcaps_metadata(self, spec: registry.DataFolder, root: Path):
        stem_list = self.collect_folder_ids(spec, root)
        stems = set(stem_list)
        sidecar = root / spec.sidecar
        path_list = sorted(sidecar.glob("*.jsonl"))
        record_list = []
        for path_item in path_list:
            with path_item.open(encoding="utf-8") as handle:
                for line_item in handle:
                    record = json.loads(line_item)
                    record_list.append(record)
        table = self.make_ifcaps_table(stems, record_list)
        return table

    # Caption record stem, by start time or unique id.
    def make_ifcaps_stem(self, record: dict, unique: dict):
        if "time" in record:
            time_text = str(record["time"])
            part_tuple = time_text.partition("-")
            seconds = float(part_tuple[0])
            start = round(seconds * 1000)
            stem = f"{record['id']}_{start}"
        else:
            record_id = str(record["id"])
            stem = unique.get(record_id, "")
        return stem

    # Join caption records onto unambiguous extracted stems.
    def make_ifcaps_table(self, stems: set, records: list):
        grouped = {}
        for stem_item in stems:
            ident, mark, offset = stem_item.rpartition("_")
            if mark and offset.isdigit():
                if ident not in grouped:
                    grouped[ident] = []
                grouped[ident].append(stem_item)
        unique = {}
        for ident, value_list in grouped.items():
            if len(value_list) == 1:
                unique[ident] = value_list[0]
        found = {}
        for record_item in records:
            caption = self.parse_ifcaps_caption(record_item["caption"])
            stem = ""
            if caption:
                stem = self.make_ifcaps_stem(record_item, unique)
            if stem in stems:
                if stem not in found:
                    found[stem] = []
                found[stem].append((("caption", (caption,)),))
        table = {}
        for stem_item, row_list in found.items():
            table[stem_item] = tuple(row_list)
        return table

    # Sidecar metadata table, read once per process.
    def load_sidecar(self, spec: registry.DataFolder, root: Path):
        if spec.sidecar not in self.sidecars:
            path = root / spec.sidecar
            if spec.sidecar_kind == "fma":
                table = self.load_fma_metadata(path)
            elif spec.sidecar_kind == "ifcaps":
                table = self.load_ifcaps_metadata(spec, root)
            elif spec.sidecar_kind == "captions":
                table = self.load_captions_metadata(path)
            else:
                table = self.load_freesound_metadata(path)
            self.sidecars[spec.sidecar] = table
        table = self.sidecars[spec.sidecar]
        return table

    # Sorted ids of every audio file under the declared root.
    def collect_folder_ids(self, spec: registry.DataFolder, root: Path):
        if spec.root not in self.listings:
            base = root / spec.root
            stem_list = []
            for path_item in base.rglob(f"*{spec.suffix}"):
                stem_list.append(path_item.stem)
            self.listings[spec.root] = sorted(stem_list)
        id_list = self.listings[spec.root]
        return id_list

    # Identity from id and byte count, never the whole file.
    def make_file_hash(self, dataset: str, ident: str, path: Path):
        stat = path.stat()
        file_hash = self.locator.make_file_hash(dataset, ident, stat.st_size)
        return file_hash

    # Total text a prompt could draw on for this clip.
    def get_metadata_chars(self, metadata: tuple):
        n_chars = 0
        for metadata_item in metadata:
            for value_item in metadata_item[1]:
                text = str(value_item)
                n_chars += len(text)
        return n_chars

    # Probed seconds of one file; None when the probe fails.
    def get_folder_seconds(self, path: Path):
        seconds = None
        try:
            path_text = str(path)
            probe = self.probe.get_audio_probe(path_text)
            seconds = -1.0
            if probe.num_frames:
                seconds = round(probe.num_frames / probe.sample_rate, 3)
        except (RuntimeError, OSError, ValueError, ZeroDivisionError) as error:
            print(f"clip skipped {path}: {error!r}")
            self.bad_item_list.append(path)
        return seconds

    # Clips of one folder id whose metadata clears the floors.
    def collect_folder_entry(
        self, scan: DataClipScan, dataset: str, spec: registry.DataFolder, ident: str,
        path: Path, entry: tuple, seconds_min: float, chars_min: int,
    ):
        is_ifcaps = spec.sidecar_kind == "ifcaps"
        entry_list = (entry,)
        if is_ifcaps:
            entry_list = entry
        kept_list = []
        for entry_idx, metadata in enumerate(entry_list):
            if self.get_metadata_chars(metadata) >= chars_min:
                kept_list.append((entry_idx, metadata))
        scan.dropped += len(entry_list) - len(kept_list)
        seconds = 0.0
        if kept_list and not is_ifcaps:
            seconds = self.get_folder_seconds(path)
            if seconds is None or seconds < seconds_min:
                scan.dropped += len(kept_list)
                kept_list = []
        for kept_item in kept_list:
            entry_idx, metadata = kept_item
            pair = ident
            if is_ifcaps:
                pair = f"{ident}:{entry_idx}"
            sample_hash = self.make_file_hash(dataset, pair, path)
            source = self.locator.make_file_locator(dataset, ident)
            clip_seconds = max(seconds, 0.0)
            clip = SampleClip(
                dataset=dataset,
                sample_hash=sample_hash,
                source=source,
                metadata=metadata,
                seconds=clip_seconds,
            )
            scan.clip_list.append(clip)

    # Clips of one pseudo shard, metadata joined by id.
    def make_folder_clips(
        self, dataset: str, spec: registry.DataFolder, shard: int, root: Path,
        seconds_min: float, chars_min: int,
    ):
        id_list = self.collect_folder_ids(spec, root)
        table = self.load_sidecar(spec, root)
        scan = DataClipScan(clip_list=[], dropped=0)
        start_idx = shard * self.cfg.folder_shard
        end_idx = start_idx + self.cfg.folder_shard
        for ident_item in id_list[start_idx:end_idx]:
            # fma pads ids on disk but not in tracks.csv
            key = ident_item
            if ident_item.isdigit():
                number = int(ident_item)
                key = str(number)
            entry = table.get(key)
            path = spec.get_path(root, ident_item)
            if entry is None:
                scan.dropped += 1
            else:
                self.collect_folder_entry(
                    scan, dataset, spec, ident_item, path, entry, seconds_min, chars_min,
                )
        return scan

    # Listing rows read once, audio-path sorted.
    def load_clip_list(self, spec: registry.DataClipList, root: Path):
        if spec.listing not in self.listings:
            path = Path(spec.listing)
            if not path.is_absolute():
                path = root / path
            row_list = []
            with path.open(encoding="utf-8") as handle:
                for line_item in handle:
                    line = line_item.strip()
                    if line:
                        row = json.loads(line)
                        audio = str(row["audio"])
                        caption = self.make_text(row.get("caption"))
                        row_list.append((audio, caption))
            self.listings[spec.listing] = sorted(row_list)
        row_list = self.listings[spec.listing]
        return row_list

    # One listing row as a clip unless audio is missing.
    def collect_list_row(
        self, scan: DataClipScan, dataset: str, spec: registry.DataClipList, root: Path,
        ident: str, caption: str, chars_min: int,
    ):
        caption_tuple = ()
        if caption:
            caption_tuple = (caption,)
        metadata = (("caption", caption_tuple),)
        n_chars = self.get_metadata_chars(metadata)
        size = None
        if n_chars < chars_min:
            scan.dropped += 1
        else:
            path = spec.get_path(root, ident)
            try:
                stat = path.stat()
                size = stat.st_size
            except OSError as error:
                print(f"clip skipped {path}: {error!r}")
                self.bad_item_list.append(path)
                scan.dropped += 1
        if size is not None:
            sample_hash = self.locator.make_list_hash(dataset, ident, size, caption)
            source = self.locator.make_file_locator(dataset, ident)
            clip = SampleClip(
                dataset=dataset,
                sample_hash=sample_hash,
                source=source,
                metadata=metadata,
                seconds=0.0,
            )
            scan.clip_list.append(clip)

    # Clips of one manifest window; missing audio dropped.
    def make_list_clips(
        self, dataset: str, spec: registry.DataClipList, shard: int, root: Path,
        seconds_min: float, chars_min: int,
    ):
        row_list = self.load_clip_list(spec, root)
        scan = DataClipScan(clip_list=[], dropped=0)
        start_idx = shard * self.cfg.folder_shard
        end_idx = start_idx + self.cfg.folder_shard
        for row_item in row_list[start_idx:end_idx]:
            ident, caption = row_item
            self.collect_list_row(scan, dataset, spec, root, ident, caption, chars_min)
        return scan

    # One stored rank-file line as a (shard, rows) pair.
    def make_manifest_pair(self, line: str, dataset_name: str):
        shard, packed = json.loads(line)
        clip_list = []
        for packed_item in packed:
            sample_hash, source, caption = packed_item
            metadata = (("caption", (caption,)),)
            clip = SampleClip(dataset_name, sample_hash, source, metadata)
            clip_list.append(clip)
        pair = (shard, tuple(clip_list))
        return pair

    # Stored (shard, rows) pairs of every rank file, shard order.
    def collect_manifest_shards(self, base: Path, dataset_name: str):
        pair_list = []
        path_list = sorted(base.glob("rank*.jsonl"))
        for path_item in path_list:
            with path_item.open(encoding="utf-8") as handle:
                for line_item in handle:
                    pair = self.make_manifest_pair(line_item, dataset_name)
                    pair_list.append(pair)
        shard_key = itemgetter(0)
        sorted_list = sorted(pair_list, key=shard_key)
        return sorted_list

    # Regroup prepared rows for this world and shard size.
    def load_cache_manifest(
        self, spec, root: Path, rank: int, world: int,
        shard_clips: int, max_files: int, max_shards: int, skip_files: int = 0,
    ):
        base = root / spec.manifest
        summary_path = base / "summary.json"
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        # the stored summary keeps its original corpus key
        pair_list = self.collect_manifest_shards(base, summary["corpus"])
        manifest = self.make_cache_manifest(
            summary, pair_list, rank, world, shard_clips, max_files, max_shards,
            skip_files,
        )
        return manifest

    # Rows repacked into consecutive shards of one size.
    def make_repacked_pairs(self, row_list: list, shard_clips: int):
        n_shards = (len(row_list) + shard_clips - 1) // shard_clips
        pair_list = []
        for shard_idx in range(n_shards):
            start_idx = shard_idx * shard_clips
            chunk = tuple(row_list[start_idx:start_idx + shard_clips])
            pair_list.append((shard_idx, chunk))
        return pair_list

    # Regroup stored shard pairs for this world and shard size.
    def make_cache_manifest(
        self, summary: dict, pairs: list, rank: int, world: int,
        shard_clips: int, max_files: int, max_shards: int, skip_files: int = 0,
    ):
        total = summary["clips"]
        if skip_files or shard_clips != summary["shard_clips"]:
            row_list = []
            for pair_item in pairs:
                row_list.extend(pair_item[1])
            row_list = row_list[skip_files:]
            total = len(row_list)
            pairs = self.make_repacked_pairs(row_list, shard_clips)
        clips = total
        if max_files:
            clips = min(total, max_files)
        shards = (clips + shard_clips - 1) // shard_clips
        if max_shards:
            shards = min(shards, max_shards)
            clips = min(clips, shards * shard_clips)
        rank_list = []
        for pair_item in pairs:
            shard, packed = pair_item
            if shard < shards and shard % world == rank:
                count = min(len(packed), clips - shard * shard_clips)
                rank_list.append((shard, packed[:count]))
        rank_shards = tuple(rank_list)
        manifest = DataManifest(summary["digest"], clips, shards, rank_shards)
        return manifest

    # Each metadata type to its values, one entry per type.
    def make_row_metadata(self, row: dict, spec: registry.DataShards):
        item_list = []
        for field_item in spec.text_fields:
            value_tuple = self.make_value_tuple(row.get(field_item))
            item_list.append((field_item, value_tuple))
        for field_item in spec.tag_fields:
            tag_tuple = self.make_tag_tuple(row.get(field_item))
            item_list.append((field_item, tag_tuple))
        items = tuple(item_list)
        return items

    # One parquet row as a clip above the text floor.
    def collect_shard_row(
        self, scan: DataClipScan, dataset: str, spec: registry.DataShards,
        shard: int, row_idx: int, row: dict, chars_min: int,
    ):
        metadata = self.make_row_metadata(row, spec)
        n_chars = self.get_metadata_chars(metadata)
        if n_chars < chars_min:
            scan.dropped += 1
        else:
            source = self.locator.make_locator(dataset, shard, row_idx)
            text_list = []
            for metadata_item in metadata:
                text_list.extend(metadata_item[1])
            row_key = source + "".join(text_list)
            key_bytes = row_key.encode()
            sha = hashlib.sha1(key_bytes)
            sample_hash = sha.hexdigest()
            clip = SampleClip(
                dataset=dataset,
                sample_hash=sample_hash,
                source=source,
                metadata=metadata,
                seconds=0.0,
            )
            scan.clip_list.append(clip)

    # Rows from text columns alone; audio bytes stay unread.
    def make_shard_clips(
        self, dataset: str, spec: registry.DataShards, shard: int, cache: Path,
        seconds_min: float, chars_min: int,
    ):
        parquet = self.shard_table.open_shard_table(spec, shard, cache)
        column_list = []
        column_list.extend(spec.text_fields)
        column_list.extend(spec.tag_fields)
        batch_iter = parquet.iter_batches(batch_size=self.cfg.parquet_batch, columns=column_list)
        scan = DataClipScan(clip_list=[], dropped=0)
        row_idx = 0
        for record_batch in batch_iter:
            for row_item in record_batch.to_pylist():
                self.collect_shard_row(scan, dataset, spec, shard, row_idx, row_item, chars_min)
                row_idx += 1
        return scan

    # Shard count, scanned for folders and declared for parquet.
    def get_version_shards(self, spec, root: Path):
        folder_shard = self.cfg.folder_shard
        if isinstance(spec, registry.DataClipList):
            row_list = self.load_clip_list(spec, root)
            n_shards = (len(row_list) + folder_shard - 1) // folder_shard
        elif isinstance(spec, registry.DataFolder):
            id_list = self.collect_folder_ids(spec, root)
            n_shards = (len(id_list) + folder_shard - 1) // folder_shard
        else:
            n_shards = spec.shards
        return n_shards

    # One shard's clips, or None past the staged parquet prefix.
    def make_version_clips(
        self, name: str, spec, shard: int, root: Path,
        seconds_min: float, chars_min: int,
    ):
        if isinstance(spec, registry.DataClipList):
            scan = self.make_list_clips(name, spec, shard, root, seconds_min, chars_min)
        elif isinstance(spec, registry.DataFolder):
            scan = self.make_folder_clips(name, spec, shard, root, seconds_min, chars_min)
        else:
            try:
                scan = self.make_shard_clips(
                    name, spec, shard, root / "hf_cache", seconds_min, chars_min
                )
            except FileNotFoundError:
                # offline cache ends here; the staged prefix is the dataset
                self.bad_item_list.append(f"{name}#{shard:05d}")
                scan = None
        return scan

    # Dataset names behind one mix name or comma list.
    def collect_dataset_names(self, datasets: str):
        name_list = []
        mix = registry.DATASET_MIXES.get(datasets)
        if mix is not None:
            for member_item in mix.members:
                name_list.append(member_item[0])
        else:
            for part_item in datasets.split(","):
                part = part_item.strip()
                if part:
                    name_list.append(part)
        return name_list

    # Fresh walk state, every dataset at its shard budget.
    def make_dataset_walk(self, name_list: list, root: Path, max_files: int):
        walk = DataWalk(
            name_list=name_list, spec_dict={}, budget_dict={}, limit_dict={},
            kept_dict={}, dropped_dict={}, passed_dict={},
        )
        for name_item in name_list:
            walk.budget_dict[name_item] = max_files
        for name_item in name_list:
            walk.spec_dict[name_item] = registry.get_version(name_item)
        for name_item in name_list:
            spec = walk.spec_dict[name_item]
            walk.limit_dict[name_item] = self.get_version_shards(spec, root)
        for name_item in name_list:
            walk.kept_dict[name_item] = []
            walk.dropped_dict[name_item] = 0
            walk.passed_dict[name_item] = 0
        return walk

    # Datasets that still have shards and budget at this shard.
    def collect_live_names(self, walk: DataWalk, shard: int):
        live_list = []
        for name_item in walk.name_list:
            budget = walk.budget_dict[name_item]
            has_room = not budget or len(walk.kept_dict[name_item]) < budget
            if shard < walk.limit_dict[name_item] and has_room:
                live_list.append(name_item)
        return live_list

    # Walk one shard of one dataset into the kept rows.
    def collect_version_shard(
        self, walk: DataWalk, name: str, shard: int, root: Path,
        seconds_min: float, chars_min: int, skip_files: int,
    ):
        spec = walk.spec_dict[name]
        scan = self.make_version_clips(name, spec, shard, root, seconds_min, chars_min)
        if scan is None:
            walk.limit_dict[name] = shard
        else:
            clip_list = scan.clip_list
            passed = walk.passed_dict[name]
            # walk past the clips an earlier slot already cached
            if passed < skip_files:
                take = min(skip_files - passed, len(clip_list))
                walk.passed_dict[name] = passed + take
                clip_list = clip_list[take:]
            budget = walk.budget_dict[name]
            room = len(clip_list)
            if budget:
                room = budget - len(walk.kept_dict[name])
            walk.kept_dict[name].extend(clip_list[:room])
            walk.dropped_dict[name] += scan.dropped

    # Round-robin one shard per version, past skip_files, until budgets fill.
    def prepare_raw_dataset(
        self, root: Path, datasets: str, max_files: int = 0, seconds_min: float = 0.0,
        chars_min: int = 1, skip_files: int = 0,
    ):
        name_list = self.collect_dataset_names(datasets)
        walk = self.make_dataset_walk(name_list, root, max_files)
        shard = 0
        live_list = self.collect_live_names(walk, shard)
        while live_list:
            for name_item in live_list:
                self.collect_version_shard(
                    walk, name_item, shard, root, seconds_min, chars_min, skip_files,
                )
            shard += 1
            live_list = self.collect_live_names(walk, shard)
        row_list = []
        for name_item in name_list:
            row_list.extend(walk.kept_dict[name_item])
        for name_item in name_list:
            n_kept = len(walk.kept_dict[name_item])
            n_dropped = walk.dropped_dict[name_item]
            print(f"dataset {name_item}: kept {n_kept}, dropped {n_dropped}, shards {shard}")
        # shard order, so one cache shard reads one parquet shard
        row_key = attrgetter("dataset", "source")
        sorted_list = sorted(row_list, key=row_key)
        return sorted_list


class LatentCacheReader:
    # Header of one finished latent cache run.
    def load_cache_header(self, cache_dir: Path):
        header_path = get_cache_header_path(cache_dir)
        header_text = header_path.read_text()
        header = json.loads(header_text)
        return header

    # Headers load for every cache; the first one returns.
    def check_cache_bundle(self, cache_dirs: list):
        first = None
        for cache_dir in cache_dirs:
            header = self.load_cache_header(cache_dir)
            if first is None:
                first = header
        return first

    # One CLI token, 'ref' or '0.4', to its blob label.
    def make_level_item(self, level: str):
        token = level.strip()
        if token == "ref":
            item = token
        else:
            # two decimals, trailing zeros dropped, one decimal kept
            value = float(token)
            text = f"{value:.2f}"
            text = text.rstrip("0")
            item = f"n{text}"
            if text.endswith("."):
                item = f"n{text}0"
        return item

    # Clip rows of one per-shard jsonl table.
    def load_clip_rows(self, clip_path: Path):
        text = clip_path.read_text()
        clip_list = []
        for line_item in text.splitlines():
            if line_item:
                clip = json.loads(line_item)
                clip_list.append(clip)
        return clip_list

    # True when one clip table may hold wavcaps audio.
    def check_wavcaps_clips(self, clip_path: Path):
        found = False
        for clip_item in self.load_clip_rows(clip_path):
            domain = get_domain(clip_item["source"])
            # list rows may point at wavcaps files by path
            if domain.startswith(("wavcaps", "list:")):
                found = True
        return found

    # True when any stored clip may hold wavcaps audio.
    def check_wavcaps_cache(self, cache_dirs: list):
        found = False
        for cache_dir in cache_dirs:
            for clip_path in get_cache_clip_paths(cache_dir):
                if not found:
                    found = self.check_wavcaps_clips(clip_path)
        return found

    # Count one stored clip, keeping it when long enough.
    def collect_clip_latent(
        self, scan: LatentScan, cache_dir: Path, level: str, min_frames: int, clip: dict,
    ):
        valid_frames = clip["valid_frames"]
        if valid_frames == 0:
            scan.corrupted_wav += 1
        elif valid_frames < min_frames:
            # short clips leave here, so no later pass rescans
            scan.short += 1
        else:
            scan.taken += 1
            gain = float(clip.get("gain", 1.0))
            phase_flip = bool(clip.get("phase_flip", False))
            start_sample = int(clip.get("start_sample", 0))
            latent = SampleLatent(
                cache_dir=str(cache_dir),
                shard=clip["shard"],
                row=clip["row"],
                level=level,
                source=clip["source"],
                valid_frames=valid_frames,
                gain=gain,
                phase_flip=phase_flip,
                start_sample=start_sample,
            )
            scan.latent_list.append(latent)

    # Stored clips of one cache level, with drop counts.
    def collect_cache_latents(self, cache_dir: Path, level: str, min_frames: int):
        cache_path = Path(cache_dir)
        scan = LatentScan(
            name=cache_path.name, latent_list=[],
            shards=0, corrupted_wav=0, short=0, taken=0,
        )
        for clip_path in get_cache_clip_paths(cache_dir):
            clip_list = self.load_clip_rows(clip_path)
            if clip_list:
                scan.shards += 1
                first_shard = clip_list[0]["shard"]
                blob = get_cache_shard_path(cache_dir, first_shard, level)
                if blob.is_file():
                    for clip_item in clip_list:
                        self.collect_clip_latent(scan, cache_dir, level, min_frames, clip_item)
                else:
                    print(f"shard skipped, {cache_dir.name} has no {blob.name}")
        return scan

    # One line of totals, naming any cache that gave nothing.
    def log_ledger(self, scan_list: list, level: str, total: int):
        empty_list = []
        n_shards = 0
        n_corrupt = 0
        n_short = 0
        for scan_item in scan_list:
            if not scan_item.taken:
                empty_list.append(scan_item.name)
            n_shards += scan_item.shards
            n_corrupt += scan_item.corrupted_wav
            n_short += scan_item.short
        note = ""
        if empty_list:
            note = f", EMPTY {empty_list}"
        print(f"latents {total} at level {level} from {len(scan_list)} caches, "
              f"{n_shards} shards, dropped {n_corrupt} corrupt_wav {n_short} short{note}")

    # True when the source names any excluded eval ytid.
    def check_excluded_source(self, source: str, exclusions: frozenset):
        excluded = False
        for ytid_item in exclusions:
            if ytid_item in source:
                excluded = True
        return excluded

    # Drop rows whose source names an excluded eval ytid.
    def split_excluded_rows(self, rows: list, exclusions: frozenset):
        kept_list = []
        for row_item in rows:
            if not self.check_excluded_source(row_item.source, exclusions):
                kept_list.append(row_item)
        return kept_list

    # One entry per stored clip at the level in use.
    def collect_latents(
        self, cache_dirs: list, level: str, min_frames: int, exclusions: frozenset,
    ):
        latent_list = []
        scan_list = []
        for cache_dir in cache_dirs:
            scan = self.collect_cache_latents(cache_dir, level, min_frames)
            latent_list.extend(scan.latent_list)
            scan_list.append(scan)
        self.log_ledger(scan_list, level, len(latent_list))
        if exclusions:
            latent_list = self.split_excluded_rows(latent_list, exclusions)
        return latent_list

    # Stable hash of one source, independent of cache order.
    def get_source_hash(self, spec, source: str, purpose: str):
        key = f"{spec.split_seed}:{purpose}:{source}"
        key_bytes = key.encode()
        sha = hashlib.sha1(key_bytes)
        hex_text = sha.hexdigest()
        source_hash = int(hex_text[:12], 16)
        return source_hash

    # Held-out sources of one domain, chosen by stable hash.
    def collect_held_sources(self, spec, domain: str, name_set: set):
        key_list = []
        for name_idx, name_item in enumerate(name_set):
            name_hash = self.get_source_hash(spec, name_item, "select")
            key_list.append((name_hash, name_idx, name_item))
        key_list.sort()
        n_keep = min(spec.test_clips_per_domain, len(key_list) // 2)
        if not n_keep:
            print(f"fit split domain {domain}: too few sources, training only")
        held_list = []
        for key_item in key_list[:n_keep]:
            held_list.append(key_item[2])
        return held_list

    # Whole sources go to the eval shard, never half.
    def split_train_eval(self, spec, rows: list):
        source_dict = {}
        for row_item in rows:
            domain = get_domain(row_item.source)
            if domain not in source_dict:
                source_dict[domain] = set()
            source_dict[domain].add(row_item.source)
        held = set()
        for domain in sorted(source_dict):
            held_list = self.collect_held_sources(spec, domain, source_dict[domain])
            held.update(held_list)
        train_shard = []
        eval_shard = []
        seen = set()
        for row_item in rows:
            if row_item.source not in held:
                train_shard.append(row_item)
            elif row_item.source not in seen:
                seen.add(row_item.source)
                eval_shard.append(row_item)
        return train_shard, eval_shard

    # Row index chunks that never mix domains inside a batch.
    def collect_samples(self, rows: list, batch_clips: int):
        domain_dict = {}
        for row_idx, row_item in enumerate(rows):
            domain = get_domain(row_item.source)
            if domain not in domain_dict:
                domain_dict[domain] = []
            domain_dict[domain].append(row_idx)
        chunk_list = []
        for domain in sorted(domain_dict):
            index_list = domain_dict[domain]
            for start_idx in range(0, len(index_list), batch_clips):
                chunk_list.append(index_list[start_idx:start_idx + batch_clips])
        return chunk_list


# One cached-latent split; an index is a seeded batch.
class DecoderLatentDataset(torch.utils.data.Dataset):
    def __init__(self, audio: ShardAudioSet, header: dict, rows: list,
                 audio_channels: int, spec: SpecTrainData,
                 item_seeds: tuple = (),
                 headers: dict | None = None):
        self.header = header
        # each cache answers for its own clip length
        self.headers = {}
        if headers:
            self.headers = dict(headers)
        # rows arrive filtered; the dataset is too big to rescan
        self.missing_audio = 0
        self.audio = audio
        self.rows = rows
        self.audio_channels = audio_channels
        self.spec = spec
        self.item_seeds = tuple(item_seeds)
        self.stride = round(header["sample_rate"] / header["frame_rate"])

    # Header of the cache this row was stored in.
    def get_header(self, row: SampleLatent):
        header = self.headers.get(row.cache_dir, self.header)
        return header

    # Waveform length matching this row's own clip length.
    def get_row_samples(self, row: SampleLatent):
        header = self.get_header(row)
        num_samples = header["frames"] * self.stride
        return num_samples

    # Batch count, one per item seed.
    def __len__(self):
        n_items = len(self.item_seeds)
        return n_items

    # Count of onsets a clip offers for one crop length.
    def get_crop_span(self, index: int, crop_frames: int):
        row = self.rows[index]
        header = self.get_header(row)
        span = row.valid_frames - crop_frames + 1
        if header.get("volume_norm"):
            # parity caches keep the upstream randint bound
            span = max(1, row.valid_frames - crop_frames)
        return span

    # Replay the frozen cache gain and flip on one crop.
    def apply_row_replay(self, wav: torch.Tensor, row: SampleLatent):
        replay = wav
        if row.gain != 1.0 or row.phase_flip:
            sign = 1.0
            if row.phase_flip:
                sign = -1.0
            scaled = wav * (row.gain * sign)  # (C, S)
            replay = scaled.clamp(-1.0, 1.0)
        return replay

    # Upstream loader drops DC and scales peak to 0.5.
    def apply_peak_norm(self, wav: torch.Tensor):
        normed = wav
        if self.spec.peak_norm:
            centred = wav - wav.mean()              # (C, S)
            magnitude = centred.abs()
            peak = magnitude.max()                  # ()
            normed = centred / (peak + 1e-8) * 0.5
        return normed

    # Per-clip waveform step after the replay; peak norm here.
    def apply_crop_wav(self, wav: torch.Tensor, valid: int, rng):
        normed = self.apply_peak_norm(wav)
        return normed

    # Cache crop origin; head crops and old caches store zero.
    def get_row_origin(self, row: SampleLatent):
        origin = row.start_sample
        return origin

    # Head-aligned crops: every clip starts at frame zero.
    def make_batch_onsets(self, rng, picks: list, crop_frames: int):
        onset_list = []
        for _ in picks:
            onset_list.append(0)
        return onset_list

    # Row indices one seeded batch draws.
    def make_batch_picks(self, rng):
        pick_list = []
        n_rows = len(self.rows)
        for _ in range(self.spec.batch_clips):
            pick = rng.randrange(n_rows)
            pick_list.append(pick)
        return pick_list

    # Draw and crop one deterministic batch entirely on the CPU.
    def __getitem__(self, index: int):
        rng = random.Random(self.item_seeds[index])
        crop_frames = self.spec.crop_frames
        pick_list = self.make_batch_picks(rng)
        onset_list = self.make_batch_onsets(rng, pick_list, crop_frames)
        latent, wav = self.load_crop_batch(pick_list, crop_frames, onset_list, rng)
        batch = VaeBatch(latent=latent, ground_truth=wav)
        return batch

    # Header of one finished latent cache run.
    @staticmethod
    def load_cache_header(cache_dir: Path):
        reader = LatentCacheReader()
        header = reader.load_cache_header(cache_dir)
        return header

    # Every cache header loads; the first one returns.
    @staticmethod
    def check_cache_bundle(cache_dirs: list, level: str = ""):
        reader = LatentCacheReader()
        first = reader.check_cache_bundle(cache_dirs)
        return first

    # One CLI token, 'ref' or '0.4', to its blob label.
    @staticmethod
    def make_level_item(level: str):
        reader = LatentCacheReader()
        item = reader.make_level_item(level)
        return item

    # True when any stored clip may hold wavcaps audio.
    @staticmethod
    def check_wavcaps_cache(cache_dirs: list):
        reader = LatentCacheReader()
        found = reader.check_wavcaps_cache(cache_dirs)
        return found

    # One entry per stored clip at the level in use.
    @staticmethod
    def collect_latents(
        cache_dirs: list, level: str, min_frames: int = 0,
        exclusions: frozenset = frozenset(),
    ):
        reader = LatentCacheReader()
        latent_list = reader.collect_latents(cache_dirs, level, min_frames, exclusions)
        return latent_list

    # Stable hash of one source, independent of cache order.
    @staticmethod
    def get_source_hash(spec, source: str, purpose: str):
        reader = LatentCacheReader()
        source_hash = reader.get_source_hash(spec, source, purpose)
        return source_hash

    # Whole sources go to the eval shard, never half.
    @staticmethod
    def split_train_eval(spec, rows: list):
        reader = LatentCacheReader()
        split = reader.split_train_eval(spec, rows)
        return split

    # Row index chunks that never mix domains inside a batch.
    @staticmethod
    def collect_samples(rows: list, batch_clips: int):
        reader = LatentCacheReader()
        chunk_list = reader.collect_samples(rows, batch_clips)
        return chunk_list

    # One [clips, latent_dim, frames] shard, read per access.
    def get_shard_tensor(self, row: SampleLatent):
        cache_dir = Path(row.cache_dir)
        shard_path = get_cache_shard_path(cache_dir, row.shard, row.level)
        shard_text = str(shard_path)
        tensor_dict = load_file(shard_text)
        latent = tensor_dict["z"]  # (N, D, T)
        return latent

    # One aligned crop, None when the clip audio vanished.
    def load_row_crop(
        self, row: SampleLatent, start: int, crop_frames: int, num_samples: int, rng,
    ):
        latent = self.get_shard_tensor(row)  # (N, D, T)
        origin = self.get_row_origin(row)
        wav, valid = self.audio.load_segment(
            row.source,
            self.header["sample_rate"],
            self.audio_channels,
            origin + start * self.stride,
            num_samples,
        )
        crop = None
        if valid == 0:
            # a clip whose audio vanished must never pass unseen
            self.missing_audio += 1
            print(f"crop dropped, no audio for {row.source} "
                  f"(total {self.missing_audio})")
        else:
            wav = self.apply_row_replay(wav, row)      # (C, S)
            wav = self.apply_crop_wav(wav, valid, rng)
            latent_crop = latent[row.row, :, start:start + crop_frames]  # (D, F)
            crop = LatentCrop(latent=latent_crop, wav=wav)
        return crop

    # Load aligned latent/waveform crops without decoding full waveforms.
    def load_crop_batch(self, picks: list, crop_frames: int, onsets: list, rng=None):
        # waveform length follows the crop; they cannot disagree
        num_samples = crop_frames * self.stride
        latent_list = []
        wav_list = []
        for pick_idx, onset_item in zip(picks, onsets):
            row = self.rows[pick_idx]
            crop = self.load_row_crop(row, onset_item, crop_frames, num_samples, rng)
            if crop is not None:
                latent_list.append(crop.latent)
                wav_list.append(crop.wav)
        latents = torch.stack(latent_list)  # (B, D, F)
        waveforms = torch.stack(wav_list)
        return latents, waveforms

    # Transfer one CPU crop and apply the pretrained latent scale.
    @staticmethod
    def load_batch_device(batch: VaeBatch, codec, device: str):
        non_blocking = False
        if device != "cpu":
            non_blocking = torch.cuda.is_available()
        # cached latents default fp16; match the decoder before its conv
        dtype = getattr(codec, "dtype", torch.float32)
        latent = batch.latent.to(device, dtype=dtype, non_blocking=non_blocking)  # (B, D, F)
        ground_truth = batch.ground_truth.to(device, non_blocking=non_blocking)
        latent = codec.apply_latent_scale(latent, "denormalize")  # (B, D, F)
        device_batch = VaeBatch(latent=latent, ground_truth=ground_truth)
        return device_batch


# Composable per-clip channel view: keep, mean, or one channel.
class ApplyChannelMix:
    def __init__(self, weights: tuple = (1.0, 1.0, 1.0)):
        # order: stereo kept, mean dual mono, one-channel dual mono
        self.weights = weights

    # Draw one channel view for this clip.
    def __call__(self, wav: torch.Tensor, valid: int, rng):
        mode_list = rng.choices(("stereo", "mean", "pick"), weights=self.weights)
        mode = mode_list[0]
        if mode == "mean":
            mono = wav.mean(dim=0, keepdim=True)  # (1, S)
            mixed = mono.repeat(wav.shape[0], 1)
        elif mode == "pick":
            channel = rng.randrange(wav.shape[0])
            mono = wav[channel:channel + 1]       # (1, S)
            mixed = mono.repeat(wav.shape[0], 1)
        else:
            mixed = wav
        return mixed


# Cached-latent crops with composable per-clip augmentations.
class LatentAugmentDataset(DecoderLatentDataset):
    def __init__(self, audio: ShardAudioSet, header: dict, rows: list,
                 audio_channels: int, spec: SpecTrainData,
                 item_seeds: tuple = (), headers: dict | None = None,
                 transforms: tuple = ()):
        super().__init__(
            audio, header, rows, audio_channels, spec,
            item_seeds=item_seeds, headers=headers,
        )
        self.transforms = tuple(transforms)

    # Each clip runs the transform chain instead of peak norm.
    def apply_crop_wav(self, wav: torch.Tensor, valid: int, rng):
        for transform_item in self.transforms:
            wav = transform_item(wav, valid, rng)  # (C, S)
        return wav
