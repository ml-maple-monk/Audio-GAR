from __future__ import annotations

import gc
import math
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import click
import torch
import torchaudio
from safetensors.torch import load_file, save as save_tensors
from tqdm import tqdm

from .src.evaluation.result_writer import EvalResultWriter
from .src.runtime.dataroot import apply_data_root, get_data_root
from .src.dataset.training_loader import (
    LatentCacheReader,
    RawHfDataset,
    get_stream_batch,
    open_batch_stream,
)
from .src.dataset.registry import (
    LATENT_DTYPES,
    CacheDrawConfig,
    CacheRunConfig,
    CacheWriterConfig,
    DataManifest,
    SampleClip,
    get_version,
    get_schedule_steps,
    get_cache_shard_path,
    make_level_rows,
    prompt_policy,
    VERBATIM_POLICY,
)
from .src.models.generator import (
    get_generator_spec,
    get_generator_steps,
    get_level_steps_fn,
    get_noise_coefficients_fn,
    load_generator,
)
from .src.models.tokenizer import load_tokenizer


@dataclass(frozen=True)
class SpecCache:
    generator: str
    corpus: str
    max_files: int
    skip_files: int
    chars_min: int
    clip_seconds: float
    cond_policy: str
    crop_policy: str
    crop_margin: float
    channel_policy: str
    volume_norm: bool
    short_policy: str
    levels: tuple[float, ...]
    shard_clips: int
    batch_clips: int
    max_shards: int
    num_steps: int
    cfg_scale: float
    seed: int
    device: str
    store_dtype: str
    determinism: bool
    vae_dtype: str
    gen_dtype: str
    compile_blocks: bool
    compile_mode: str
    prompt_policy: str
    noise_schedule: str
    peak_norm: bool


@dataclass(frozen=True)
class FrameGeometry:
    num_samples: int
    frames: int


@dataclass(frozen=True)
class SampleDraw:
    fold: str
    kind: str
    start: int


@dataclass(frozen=True)
class SampleGain:
    gain: float = 1.0
    flip: bool = False


@dataclass
class SampleWindow:
    wav: torch.Tensor | None
    valid_frames: int
    gain: SampleGain = field(default_factory=SampleGain)


@dataclass
class ShardRequestSet:
    shard_draw_dict: dict
    request_list: list


@dataclass
class ShardEncodeState:
    kept_list: list = field(default_factory=list)
    draw_list: list = field(default_factory=list)
    valid_list: list = field(default_factory=list)
    gain_list: list = field(default_factory=list)
    store_list: list = field(default_factory=list)
    batch_row_list: list = field(default_factory=list)
    batch_wav_list: list = field(default_factory=list)
    z_store: torch.Tensor | None = None


@dataclass
class ShardInput:
    z: torch.Tensor
    eps: torch.Tensor
    prompt_list: list
    pair_list: list
    hash_list: list


@dataclass
class CacheClipRow:
    dataset: str
    sample_hash: str
    source: str
    # the prompt that conditioned this clip, per SAO 3.2
    prompt: str
    shard: int
    row: int
    # frames backed by real audio; the loop fills the rest
    valid_frames: int
    start_sample: int
    seconds_start: float
    seconds_total: float
    gain: float
    phase_flip: bool


@dataclass
class CacheHeader:
    generator: str
    sample_rate: int
    frame_rate: float
    frames: int
    volume_norm: bool


@dataclass
class CacheSummary:
    generator: str
    shards: int
    clips: int


@dataclass
class ShardLevelEvent:
    shard: int
    level: float


class CacheFamily(ABC):
    def __init__(self, spec: SpecCache, draw_cfg: CacheDrawConfig):
        self.spec = spec
        self.draw_cfg = draw_cfg

    # Load this family's codec in eval mode.
    @abstractmethod
    def open_codec(self):
        ...

    # Load this family's generator with its runtime switches.
    @abstractmethod
    def open_generator(self):
        ...

    # Sampler steps one noise level walks.
    @abstractmethod
    def get_steps(self, level: float):
        ...

    # Conditioning for one padded batch of prompts and timings.
    @abstractmethod
    def make_conditioning(self, gen, prompt_list: list, pair_list: list):
        ...

    # Timing pairs the generator wrapper honours.
    @abstractmethod
    def apply_pairs(self, pair_list: list):
        ...

    # Refuse level conventions this family never trained on.
    @abstractmethod
    def check_norm(self):
        ...


class CacheAudioxFamily(CacheFamily):
    # AudioX codec takes determinism and its VAE dtype.
    def open_codec(self):
        spec = self.spec
        codec = load_tokenizer(
            spec.generator, device=spec.device, determinism=spec.determinism,
            vae_dtype=spec.vae_dtype)
        codec.eval()
        return codec

    # AudioX DiT; auto compile takes the ladder winner unless --compile_blocks.
    def open_generator(self):
        spec = self.spec
        compile_mode = spec.compile_mode
        if compile_mode == "auto":
            compile_mode = self.draw_cfg.auto_compile_mode
            if spec.compile_blocks:
                compile_mode = "blocks"
        gen = load_generator(
            spec.generator, device=spec.device, determinism=spec.determinism,
            vae_dtype=spec.vae_dtype, compile_mode=compile_mode)
        return gen

    # AudioX truncates the schedule, so every level shares a stride.
    def get_steps(self, level: float):
        steps = get_schedule_steps(self.spec.num_steps, level)
        return steps

    # AudioX sizes its zero audio from one scalar pair.
    def make_conditioning(self, gen, prompt_list: list, pair_list: list):
        first_pair = pair_list[0]
        cond = gen.make_conditioning(
            prompt_list, seconds_start=first_pair[0], seconds_total=first_pair[1])
        return cond

    # AudioX takes the first timing pair of every batch.
    def apply_pairs(self, pair_list: list):
        applied_list = []
        n_pairs = len(pair_list)
        for start_idx in range(0, n_pairs, self.spec.batch_clips):
            batch_pair_list = pair_list[start_idx:start_idx + self.spec.batch_clips]
            first_pair = batch_pair_list[0]
            for pair_item in batch_pair_list:
                applied_list.append(first_pair)
        return applied_list

    # Peak norm is a tango convention AudioX never saw.
    def check_norm(self):
        if self.spec.peak_norm:
            raise ValueError(f"peak_norm needs a tango generator, got {self.spec.generator}")


class CacheTangoMusicFamily(CacheFamily):
    # Tango-music codec takes determinism only.
    def open_codec(self):
        spec = self.spec
        codec = load_tokenizer(spec.generator, device=spec.device, determinism=spec.determinism)
        codec.eval()
        return codec

    # Tango-music UNet takes determinism, block compile and dtype.
    def open_generator(self):
        spec = self.spec
        gen = load_generator(
            spec.generator, device=spec.device, determinism=spec.determinism,
            compile_blocks=spec.compile_blocks, gen_dtype=spec.gen_dtype)
        return gen

    # Tango-music walks the full grid; the suffix picks the level.
    def get_steps(self, level: float):
        steps = self.spec.num_steps
        return steps

    # Tango-music takes one timing pair per clip.
    def make_conditioning(self, gen, prompt_list: list, pair_list: list):
        start_list = []
        total_list = []
        for pair_item in pair_list:
            start_list.append(pair_item[0])
            total_list.append(pair_item[1])
        cond = gen.make_conditioning(
            prompt_list, seconds_start=start_list, seconds_total=total_list)
        return cond

    # Tango-music honours every clip's own timing pair.
    def apply_pairs(self, pair_list: list):
        applied_list = pair_list
        return applied_list

    # Peak norm and LUFS norm are rival level conventions.
    def check_norm(self):
        if self.spec.peak_norm and self.spec.volume_norm:
            raise ValueError("peak_norm and volume_norm are rival level conventions")


class CacheGeometry:
    def __init__(self, spec: SpecCache):
        self.spec = spec

    # Hop-aligned samples and frames of one cached clip.
    def get_frames(self, codec):
        frame_rate = getattr(codec, "frame_rate", None)
        if frame_rate is None:
            frame_rate = codec.sample_rate / codec.downsampling_ratio
        hop = round(codec.sample_rate / frame_rate)
        total = int(self.spec.clip_seconds * codec.sample_rate)
        num_samples = total // hop * hop
        geometry = FrameGeometry(num_samples=num_samples, frames=num_samples // hop)
        return geometry

    # Torch dtype the latents are stored in.
    def get_dtype(self):
        store_dtype = LATENT_DTYPES[self.spec.store_dtype]
        return store_dtype

    # Per-clip sampler seeds from content hashes, batch independent.
    def get_seeds(self, hash_list: list):
        seed_list = []
        for hash_item in hash_list:
            clip_seed = RawHfDataset.make_eps_seed(hash_item, self.spec.seed)
            seed_list.append(clip_seed)
        return seed_list

    # The draw_idx-th BF16 draw of each clip's own generator.
    def make_noise(self, shape: tuple, hash_list: list, draw_idx: int):
        if hash_list:
            seed_list = self.get_seeds(hash_list)
            draw_list = []
            for seed_item in seed_list:
                gen_stream = torch.Generator()
                gen_stream.manual_seed(seed_item)
                for step_idx in range(draw_idx + 1):
                    draw = torch.randn(shape, generator=gen_stream)  # (D, F)
                draw_list.append(draw)
            stacked = torch.stack(draw_list)  # (B, D, F)
            noise = stacked.bfloat16()
        else:
            empty_shape = (0,) + shape
            noise = torch.zeros(empty_shape, dtype=torch.bfloat16)  # (0, D, F)
        return noise


class ShardEncoder:
    def __init__(
        self, codec, spec: SpecCache, draw_cfg: CacheDrawConfig, family: CacheFamily,
        geometry: CacheGeometry, writer: EvalResultWriter,
    ):
        self.codec = codec
        self.spec = spec
        self.draw_cfg = draw_cfg
        self.family = family
        self.geometry = geometry
        self.writer = writer

    # One weighted arm of a 'name' or 'name:w,name:w' mix.
    def get_arm(self, pick_stream: random.Random, text: str):
        name_list = []
        weight_list = []
        part_list = text.split(",")
        for part_item in part_list:
            stripped = part_item.strip()
            name, mark, weight_text = stripped.partition(":")
            weight = 1.0
            if mark:
                weight = float(weight_text)
            name_list.append(name)
            weight_list.append(weight)
        choice_list = pick_stream.choices(name_list, weights=weight_list)
        arm = choice_list[0]
        return arm

    # Deterministic random-crop start; negative means no legal window.
    def get_start(self, row: SampleClip, sample_rate: int, num_samples: int):
        margin = int(self.spec.crop_margin * sample_rate)
        top = int(row.seconds * sample_rate) - num_samples - margin
        start = -1
        if top >= margin:
            crop_seed = RawHfDataset.make_crop_seed(row.sample_hash, self.spec.seed)
            crop_stream = random.Random(crop_seed)
            start = crop_stream.randint(margin, top)
        return start

    # Per-clip fold, timing arm and crop start; None drops.
    def make_draw(self, row: SampleClip, sample_rate: int, num_samples: int):
        crop_seed = RawHfDataset.make_crop_seed(row.sample_hash, self.spec.seed)
        pick_stream = random.Random(crop_seed ^ self.draw_cfg.pick_salt)
        fold = self.get_arm(pick_stream, self.spec.channel_policy)
        kind = self.get_arm(pick_stream, self.spec.cond_policy)
        crop = self.get_arm(pick_stream, self.spec.crop_policy)
        start = 0
        if crop == "random":
            start = self.get_start(row, sample_rate, num_samples)
        draw = SampleDraw(fold=fold, kind=kind, start=start)
        if start < 0:
            # no legal window; the repeat policy loops the head
            draw = None
            if self.spec.short_policy == "repeat":
                draw = SampleDraw(fold=fold, kind=kind, start=0)
        return draw

    # Per-shard draws plus the reader's load requests.
    def make_requests(self, pending_list: list, sample_rate: int, num_samples: int):
        shard_draw_dict = {}
        request_list = []
        for shard_idx, row_tuple in pending_list:
            draw_list = []
            group_list = []
            for row_item in row_tuple:
                draw = self.make_draw(row_item, sample_rate, num_samples)
                draw_list.append(draw)
                start = 0
                fold = "stereo"
                if draw is not None:
                    start = draw.start
                    fold = draw.fold
                group_list.append((row_item.source, start, fold))
            shard_draw_dict[shard_idx] = draw_list
            request_list.append(group_list)
        request_set = ShardRequestSet(shard_draw_dict=shard_draw_dict, request_list=request_list)
        return request_set

    # Full-window wav and true frames; short clips loop or drop.
    def pad_window(self, wav: torch.Tensor, valid: int, num_samples: int, hop: int):
        window = SampleWindow(wav=None, valid_frames=0)
        is_short = valid < num_samples
        is_dropped = is_short and (self.spec.short_policy == "drop" or valid < hop)
        if not is_dropped:
            out_wav = wav  # (C, S)
            if is_short:
                reps = -(-num_samples // valid)
                head = wav[:, :valid]  # (C, V)
                tiled = head.repeat(1, reps)
                out_wav = tiled[:, :num_samples]  # (C, S)
            window = SampleWindow(wav=out_wav, valid_frames=valid // hop)
        return window

    # Upstream loader drops DC and scales peak to 0.5.
    def apply_peak(self, wav: torch.Tensor):
        mean = wav.mean()  # ()
        centered = wav - mean
        abs_wav = centered.abs()
        peak = abs_wav.max()  # ()
        normed = centered / (peak + 1e-8) * 0.5
        return normed

    # Dual-mono windows measure as one channel, like upstream.
    def get_mono(self, wav: torch.Tensor):
        measure_wav = wav  # (C, S)
        if wav.shape[0] > 1:
            first = wav[:1]  # (1, S)
            expanded = first.expand_as(wav)
            if torch.equal(expanded, wav):
                measure_wav = first
        return measure_wav

    # One frozen LUFS gain and phase flip per clip.
    def make_gain(self, row: SampleClip, wav: torch.Tensor, loudness):
        crop_seed = RawHfDataset.make_crop_seed(row.sample_hash, self.spec.seed)
        gain_stream = random.Random(crop_seed ^ self.draw_cfg.gain_salt)
        flip_draw = gain_stream.random()
        flip = flip_draw < 0.5
        power = wav.pow(2)  # (C, S)
        energy_tensor = power.mean()
        energy = float(energy_tensor)
        gain = 1.0
        if not energy < self.draw_cfg.energy_floor:
            jitter = self.draw_cfg.lufs_jitter
            offset = gain_stream.uniform(-jitter, jitter)
            target = self.draw_cfg.lufs_target + offset
            measure_wav = self.get_mono(wav)  # (C, S)
            lufs_tensor = loudness(measure_wav)
            lufs = float(lufs_tensor)
            gain = float(10.0 ** ((target - lufs) / 20.0))
            abs_wav = wav.abs()  # (C, S)
            peak_tensor = abs_wav.max()
            peak = float(peak_tensor) * gain
            if peak > 1.0:
                # upstream declip scales an over-unity peak to 0.95
                gain *= 0.95 / peak
        clip_gain = SampleGain(gain=gain, flip=flip)
        return clip_gain

    # Gained, sign-flipped, clamped copy of one window.
    def apply_gain(self, wav: torch.Tensor, clip_gain: SampleGain):
        sign = 1.0
        if clip_gain.flip:
            sign = -1.0
        signed = clip_gain.gain * sign
        scaled = wav * signed  # (C, S)
        clamped = scaled.clamp(-1.0, 1.0)
        return clamped

    # Peak level in dBFS across every channel.
    def get_dbmax(self, wav: torch.Tensor):
        abs_wav = wav.abs()  # (C, S)
        peak = abs_wav.max()
        level = torch.log10(peak + 1e-10)  # ()
        dbmax = float(20.0 * level)
        return dbmax

    # Peak norm, then LUFS gain; a silent window drops.
    def apply_norms(self, row: SampleClip, window: SampleWindow, loudness):
        out_wav = window.wav  # (C, S)
        if self.spec.peak_norm:
            out_wav = self.apply_peak(out_wav)
        clip_gain = SampleGain()
        if loudness is not None:
            clip_gain = self.make_gain(row, out_wav, loudness)
            out_wav = self.apply_gain(out_wav, clip_gain)  # (C, S)
            dbmax = self.get_dbmax(out_wav)
            if dbmax < self.draw_cfg.silence_dbmax:
                print(f"clip dropped, silent window for {row.source}")
                out_wav = None
        normed = SampleWindow(wav=out_wav, valid_frames=window.valid_frames, gain=clip_gain)
        return normed

    # One clip's kept window, or a window without wav.
    def make_window(
        self, row: SampleClip, draw: SampleDraw | None, wav: torch.Tensor, valid: int,
        frame_geometry: FrameGeometry, loudness,
    ):
        hop = frame_geometry.num_samples // frame_geometry.frames
        window = SampleWindow(wav=None, valid_frames=0)
        if draw is not None:
            window = self.pad_window(wav, valid, frame_geometry.num_samples, hop)
        if window.wav is None:
            print(f"clip dropped, no full-window crop for {row.source}")
        else:
            window = self.apply_norms(row, window, loudness)
        return window

    # Encode one kept batch with posterior draws, never the mean.
    def encode_batch(self, state: ShardEncodeState):
        frame_geometry = self.geometry.get_frames(self.codec)
        shape = (self.codec.model.spec.latent_dim, frame_geometry.frames)
        hash_list = []
        for row_item in state.batch_row_list:
            hash_list.append(row_item.sample_hash)
        eps = self.geometry.make_noise(shape, hash_list, 0)  # (B, D, F)
        wav = torch.stack(state.batch_wav_list)
        # reader yields CPU wavs; codec may be CUDA
        wav = wav.to(self.codec.device)
        z = self.codec.encode_latent_sample(wav, eps)
        z = z.to(dtype=torch.bfloat16, device="cpu")  # (B, D, F)
        store_dtype = self.geometry.get_dtype()
        z_store = z.to(store_dtype)
        state.store_list.append(z_store)
        state.batch_row_list = []
        state.batch_wav_list = []

    # Record one kept clip; a full batch encodes at once.
    def append_sample(
        self, state: ShardEncodeState, row: SampleClip, draw: SampleDraw,
        window: SampleWindow, frames: int,
    ):
        state.kept_list.append(row)
        state.draw_list.append(draw)
        valid_frames = min(window.valid_frames, frames)
        state.valid_list.append(valid_frames)
        state.gain_list.append(window.gain)
        state.batch_row_list.append(row)
        state.batch_wav_list.append(window.wav)
        if len(state.batch_row_list) == self.spec.batch_clips:
            self.encode_batch(state)

    # Encode full-window clips only; the rest of the shard drops.
    def encode_shard(self, reader, stream, shard_row_tuple: tuple, draw_list: list):
        frame_geometry = self.geometry.get_frames(self.codec)
        loudness = None
        if self.spec.volume_norm:
            loudness = torchaudio.transforms.Loudness(self.codec.sample_rate)
        state = ShardEncodeState()
        n_rows = len(shard_row_tuple)
        for start_idx in range(0, n_rows, self.spec.batch_clips):
            row_slice = shard_row_tuple[start_idx:start_idx + self.spec.batch_clips]
            wav_batch, valid_list = get_stream_batch(reader, stream)  # (B, C, S)
            for row_idx, row_item in enumerate(row_slice):
                draw = draw_list[start_idx + row_idx]
                window = self.make_window(
                    row_item, draw, wav_batch[row_idx], valid_list[row_idx],
                    frame_geometry, loudness,
                )
                if window.wav is not None:
                    self.append_sample(state, row_item, draw, window, frame_geometry.frames)
        if state.batch_row_list:
            self.encode_batch(state)
        if state.kept_list:
            state.z_store = torch.cat(state.store_list)  # (N, D, F)
        else:
            latent_dim = self.codec.model.spec.latent_dim
            empty = torch.zeros(0, latent_dim, frame_geometry.frames)  # (0, D, F)
            store_dtype = self.geometry.get_dtype()
            state.z_store = empty.to(store_dtype)
        return state

    # Per-clip timing pair the drawn arm dictates.
    def make_seconds(self, row_list: list, draw_list: list, sample_rate: int):
        pair_list = []
        for row_item, draw_item in zip(row_list, draw_list, strict=True):
            if draw_item.kind == "window":
                clip_seconds = float(self.spec.clip_seconds)
                pair = (0.0, clip_seconds)
            elif draw_item.kind == "decode_window":
                ceil_seconds = math.ceil(self.spec.clip_seconds)
                total_seconds = float(ceil_seconds)
                pair = (0.0, total_seconds)
            else:
                # upstream floors the start and ceils the source length
                ceil_seconds = math.ceil(row_item.seconds)
                start_seconds = float(draw_item.start // sample_rate)
                total_seconds = float(ceil_seconds)
                pair = (start_seconds, total_seconds)
            pair_list.append(pair)
        return pair_list

    # Clip table rows of one encoded shard.
    def make_rows(self, shard_idx: int, state: ShardEncodeState, pair_list: list):
        packed = zip(
            state.kept_list, state.draw_list, pair_list, state.valid_list, state.gain_list,
            strict=True,
        )
        row_list = []
        for row_idx, (row_item, draw_item, pair_item, valid_item, gain_item) in enumerate(packed):
            prompt_seed = RawHfDataset.make_prompt_seed(row_item.sample_hash, self.spec.seed)
            prompt = RawHfDataset.make_prompt(
                row_item.dataset, row_item.metadata, prompt_seed, self.spec.prompt_policy)
            clip_row = CacheClipRow(
                dataset=row_item.dataset, sample_hash=row_item.sample_hash,
                source=row_item.source, prompt=prompt, shard=shard_idx, row=row_idx,
                valid_frames=valid_item, start_sample=draw_item.start,
                seconds_start=pair_item[0], seconds_total=pair_item[1],
                gain=gain_item.gain, phase_flip=gain_item.flip,
            )
            row_dict = asdict(clip_row)
            row_list.append(row_dict)
        return row_list

    # Encode every shard; write its latents and clip table.
    def encode_shards(self, pending_list: list):
        frame_geometry = self.geometry.get_frames(self.codec)
        num_samples = frame_geometry.num_samples
        sample_rate = self.codec.sample_rate
        request_set = self.make_requests(pending_list, sample_rate, num_samples)
        data_root = get_data_root()
        reader, stream = open_batch_stream(
            data_root, request_set.request_list,
            sample_rate, self.codec.audio_channels, num_samples, self.spec.batch_clips,
        )
        shard_iter = tqdm(pending_list, unit="encoded shard")
        for shard_idx, shard_row_tuple in shard_iter:
            draw_list = request_set.shard_draw_dict[shard_idx]
            state = self.encode_shard(reader, stream, shard_row_tuple, draw_list)
            pair_list = self.make_seconds(state.kept_list, state.draw_list, sample_rate)
            applied_list = self.family.apply_pairs(pair_list)
            row_list = self.make_rows(shard_idx, state, applied_list)
            blob = save_tensors({"z": state.z_store})
            self.writer.write("latent_shard", blob, item=f"shard{shard_idx:06d}_ref")
            self.writer.write("latent_clip", row_list, item=f"shard{shard_idx:06d}")


class ShardDenoiser:
    def __init__(
        self, gen, spec: SpecCache, family: CacheFamily, geometry: CacheGeometry,
        writer: EvalResultWriter, clip_reader: LatentCacheReader,
    ):
        self.gen = gen
        self.spec = spec
        self.family = family
        self.geometry = geometry
        self.writer = writer
        self.clip_reader = clip_reader

    # Level item name, full precision when one decimal rounds.
    def get_item(self, shard_idx: int, level: float):
        text = f"{level:.1f}"
        rounded = float(text)
        if rounded != level:
            text = repr(level)
        item = f"shard{shard_idx:06d}_n{text}"
        return item

    # Reload one encoded shard and recreate its injection noise.
    def load_shard(self, shard_idx: int):
        latent_path = get_cache_shard_path(self.writer.root, shard_idx, "ref")
        latent_text = str(latent_path)
        tensor_dict = load_file(latent_text)
        key_set = set(tensor_dict)
        if key_set != {"z"}:
            raise ValueError(f"{latent_path} must contain only tensor 'z'")
        z = tensor_dict["z"]
        z = z.to(torch.bfloat16)  # (N, D, F)
        clip_path = self.writer.get_artifact_path("latent_clip", f"shard{shard_idx:06d}")
        clip_row_list = self.clip_reader.load_clip_rows(clip_path)
        n_clips = z.shape[0]
        n_rows = len(clip_row_list)
        if n_rows != n_clips:
            raise ValueError(f"shard {shard_idx} has {n_clips} latents but {n_rows} clip rows")
        prompt_list = []
        pair_list = []
        hash_list = []
        for row_item in clip_row_list:
            prompt_list.append(row_item["prompt"])
            seconds_start = float(row_item["seconds_start"])
            seconds_total = float(row_item["seconds_total"])
            pair_list.append((seconds_start, seconds_total))
            hash_list.append(row_item["sample_hash"])
        shape = tuple(z.shape[1:])
        eps = self.geometry.make_noise(shape, hash_list, 1)  # (N, D, F)
        shard_input = ShardInput(
            z=z, eps=eps, prompt_list=prompt_list, pair_list=pair_list, hash_list=hash_list)
        return shard_input

    # One conditioning per batch slice, reused by every level.
    def prepare_conditioning(self, prompt_list: list, pair_list: list):
        batch_clips = self.spec.batch_clips
        cond_list = []
        n_prompts = len(prompt_list)
        for start_idx in range(0, n_prompts, batch_clips):
            batch_prompt_list = prompt_list[start_idx:start_idx + batch_clips]
            batch_pair_list = pair_list[start_idx:start_idx + batch_clips]
            # tail entries repeat so compiled shapes never change
            n_pad = batch_clips - len(batch_prompt_list)
            last_prompt = batch_prompt_list[-1]
            last_pair = batch_pair_list[-1]
            for pad_idx in range(n_pad):
                batch_prompt_list.append(last_prompt)
                batch_pair_list.append(last_pair)
            cond = self.family.make_conditioning(self.gen, batch_prompt_list, batch_pair_list)
            cond_list.append(cond)
        return cond_list

    # Denoise clips in batch slices at one noise level.
    def denoise_batch(
        self, z_all: torch.Tensor, eps_all: torch.Tensor, hash_list: list, level: float,
        cond_list: list,
    ):
        coeffs_fn = get_noise_coefficients_fn(self.spec.generator, self.spec.noise_schedule)
        a_t, b_t = coeffs_fn(level)
        steps = self.family.get_steps(level)
        store_dtype = self.geometry.get_dtype()
        batch_clips = self.spec.batch_clips
        part_list = []
        n_clips = z_all.shape[0]
        start_range = range(0, n_clips, batch_clips)
        for batch_idx, start_idx in enumerate(start_range):
            stop_idx = start_idx + batch_clips
            z = z_all[start_idx:stop_idx]  # (B, D, F)
            eps = eps_all[start_idx:stop_idx]
            seed_list = self.geometry.get_seeds(hash_list[start_idx:stop_idx])
            count = z.shape[0]
            pad = batch_clips - count
            if pad > 0:
                # repeat the last latent so compile skips a recompile
                z_last = z[-1:]
                z_tail = z_last.repeat(pad, 1, 1)
                z = torch.cat([z, z_tail])  # (B, D, F)
                eps_last = eps[-1:]
                eps_tail = eps_last.repeat(pad, 1, 1)  # (P, D, F)
                eps = torch.cat([eps, eps_tail])
                for pad_idx in range(pad):
                    seed_list.append(seed_list[-1])
            z = z.to(self.spec.device)  # (B, D, F)
            eps = eps.to(self.spec.device)
            z_t = a_t * z + b_t * eps  # (B, D, F)
            x0 = self.gen.denoise_latent_loop(
                z_t, a_t, b_t, cond_list[batch_idx], steps, seed_list, self.spec.cfg_scale)
            head = x0[:count]  # (N, D, F)
            stored = head.to(store_dtype)
            part = stored.cpu()  # (N, D, F)
            part_list.append(part)
        z_out = torch.cat(part_list)
        return z_out

    # One level, one artifact, written before the next level starts.
    def write_level(
        self, shard_idx: int, shard_input: ShardInput, level: float, cond_list: list,
    ):
        item = self.get_item(shard_idx, level)
        n_clips = shard_input.z.shape[0]
        if level == 0.0 or n_clips == 0:
            # level zero is the posterior sample, sampler adds nothing
            store_dtype = self.geometry.get_dtype()
            z_out = shard_input.z.to(store_dtype)  # (N, D, F)
        else:
            # same slices the encode pass used, so batch composition holds
            z_out = self.denoise_batch(
                shard_input.z, shard_input.eps, shard_input.hash_list, level, cond_list)
        event = ShardLevelEvent(shard=shard_idx, level=level)
        blob = save_tensors({"z": z_out})
        self.writer.write("latent_shard", blob, item=item)
        event_dict = asdict(event)
        self.writer.log(event_dict)

    # Denoise one shard at every level, in level order.
    def denoise_sample(self, shard_idx: int, shard_input: ShardInput):
        has_sampler_level = False
        for level_item in self.spec.levels:
            has_sampler_level = has_sampler_level or level_item > 0.0
        cond_list = []
        n_clips = shard_input.z.shape[0]
        if self.gen is not None and n_clips and has_sampler_level:
            # real prompts, so guidance has a direction to push along
            cond_list = self.prepare_conditioning(shard_input.prompt_list, shard_input.pair_list)
        for level_item in self.spec.levels:
            self.write_level(shard_idx, shard_input, level_item, cond_list)


class CacheRunPlanner:
    def __init__(self, cfg: CacheRunConfig, draw_cfg: CacheDrawConfig):
        self.cfg = cfg
        self.draw_cfg = draw_cfg

    # Frozen cache knobs from the same-named CLI options.
    def make_spec(self):
        cfg_dict = asdict(self.cfg)
        value_dict = {}
        for spec_field in fields(SpecCache):
            if spec_field.name in cfg_dict:
                value_dict[spec_field.name] = cfg_dict[spec_field.name]
        level_list = []
        level_text_list = self.cfg.noise_levels.split(",")
        for level_text in level_text_list:
            level_value = float(level_text)
            level_list.append(level_value)
        value_dict["levels"] = tuple(level_list)
        if not self.cfg.num_steps:
            value_dict["num_steps"] = get_generator_steps(self.cfg.generator)
        spec = SpecCache(**value_dict)
        return spec

    # The variant owning every generator-specific cache step.
    def make_family(self, spec: SpecCache):
        if spec.generator.startswith("audiox"):
            family = CacheAudioxFamily(spec, self.draw_cfg)
        elif spec.generator.startswith("tango-music"):
            family = CacheTangoMusicFamily(spec, self.draw_cfg)
        else:
            raise ValueError(f"latent cache runs audiox or tango-music, got {spec.generator}")
        return family

    # Family-only flags, refused before any corpus work.
    def check_spec(self, spec: SpecCache):
        family = self.make_family(spec)
        get_noise_coefficients_fn(spec.generator, spec.noise_schedule)
        family.check_norm()

    # Raw corpus split into fixed-size shards.
    def make_corpus(self, loader: RawHfDataset, spec: SpecCache):
        # margins shrink the crop range, so the floor grows
        seconds_min = spec.clip_seconds + 2 * spec.crop_margin
        if spec.short_policy == "repeat":
            seconds_min = 0.0
        data_root = get_data_root()
        row_list = loader.prepare_raw_dataset(
            data_root, spec.corpus, spec.max_files, seconds_min,
            spec.chars_min, skip_files=spec.skip_files,
        )
        n_rows = len(row_list)
        total = (n_rows + spec.shard_clips - 1) // spec.shard_clips
        if spec.max_shards:
            total = min(total, spec.max_shards)
        shard_list = []
        for shard_idx in range(total):
            start_idx = shard_idx * spec.shard_clips
            row_slice = row_list[start_idx:start_idx + spec.shard_clips]
            row_tuple = tuple(row_slice)
            shard_list.append((shard_idx, row_tuple))
        clips = min(n_rows, total * spec.shard_clips)
        rank_shards = tuple(shard_list)
        # nothing reads a digest, so the raw corpus carries none
        corpus = DataManifest(digest="", clips=clips, shards=total, rank_shards=rank_shards)
        return corpus

    # Load prepared shards or build small corpora.
    def load_corpus(self, spec: SpecCache):
        loader = RawHfDataset()
        version = None
        if "," not in spec.corpus:
            version = get_version(spec.corpus)
        manifest = getattr(version, "manifest", "")
        if version is not None and manifest:
            data_root = get_data_root()
            corpus = loader.load_cache_manifest(
                version, data_root, 0, 1, spec.shard_clips,
                spec.max_files, spec.max_shards, spec.skip_files,
            )
        else:
            corpus = self.make_corpus(loader, spec)
        return corpus


class LatentCacheRun:
    def __init__(
        self, cfg: CacheRunConfig, draw_cfg: CacheDrawConfig, spec: SpecCache,
        family: CacheFamily, corpus: DataManifest, writer: EvalResultWriter,
    ):
        self.cfg = cfg
        self.draw_cfg = draw_cfg
        self.spec = spec
        self.family = family
        self.corpus = corpus
        self.writer = writer
        self.geometry = CacheGeometry(spec)
        self.clip_reader = LatentCacheReader()

    # Resolve the spec, corpus and writer.
    @classmethod
    def make_from_config(cls, cfg: CacheRunConfig, draw_cfg: CacheDrawConfig):
        apply_data_root(cfg.data_root)
        planner = CacheRunPlanner(cfg, draw_cfg)
        spec = planner.make_spec()
        planner.check_spec(spec)
        corpus = planner.load_corpus(spec)
        writer = EvalResultWriter.make_from_dir(CacheWriterConfig(), cfg.out_dir)
        if cfg.exec_batch_clips:
            # runtime batch only; artifacts stay batch-independent by seeding
            spec = replace(spec, batch_clips=cfg.exec_batch_clips)
        family = planner.make_family(spec)
        cache_run = cls(cfg, draw_cfg, spec, family, corpus, writer)
        return cache_run

    # Encode every shard, then free the codec.
    def encode_corpus(self):
        shard_list = self.corpus.rank_shards
        if shard_list:
            codec = self.family.open_codec()
            encoder = ShardEncoder(
                codec, self.spec, self.draw_cfg, self.family, self.geometry, self.writer)
            encoder.encode_shards(shard_list)
            # the codec must not sit beside the DiT
            del codec, encoder
            gc.collect()
            if torch.cuda.is_available() and self.spec.device.startswith("cuda"):
                torch.cuda.empty_cache()

    # Generator only when a sampler level has shards to run.
    def open_generator(self):
        has_sampler_level = False
        for level_item in self.spec.levels:
            has_sampler_level = has_sampler_level or level_item > 0.0
        gen = None
        if has_sampler_level and self.corpus.rank_shards:
            gen = self.family.open_generator()
        return gen

    # Denoise every shard at every level.
    def denoise_shards(self):
        gen = self.open_generator()
        denoiser = ShardDenoiser(
            gen, self.spec, self.family, self.geometry, self.writer, self.clip_reader)
        shard_iter = tqdm(self.corpus.rank_shards, unit="shard")
        for shard_item in shard_iter:
            shard_idx = shard_item[0]
            shard_input = denoiser.load_shard(shard_idx)
            denoiser.denoise_sample(shard_idx, shard_input)

    # Header, level table and run summary.
    def finish_cache(self):
        spec = self.spec
        model_spec = get_generator_spec(spec.generator)
        frame_geometry = self.geometry.get_frames(model_spec)
        steps_fn = get_level_steps_fn(spec.generator, spec.noise_schedule)
        coeffs_fn = get_noise_coefficients_fn(spec.generator, spec.noise_schedule)
        level_rows = make_level_rows(spec.levels, spec.num_steps, steps_fn, coeffs_fn)
        self.writer.write("latent_level", level_rows)
        header = CacheHeader(
            generator=spec.generator, sample_rate=model_spec.sample_rate,
            frame_rate=model_spec.sample_rate / model_spec.downsampling_ratio,
            frames=frame_geometry.frames, volume_norm=spec.volume_norm,
        )
        header_dict = asdict(header)
        self.writer.write("latent_cache", header_dict)
        summary = CacheSummary(
            generator=spec.generator, shards=self.corpus.shards, clips=self.corpus.clips,
        )
        summary_dict = asdict(summary)
        self.writer.finish(summary_dict)

    # Encode, denoise every level, then write the header.
    def run(self):
        self.encode_corpus()
        self.denoise_shards()
        self.finish_cache()


@click.command(help="Cache one-step denoised generator latents per noise level.")
@click.option("--data_root", type=Path, required=True, help="corpus, weights and models")
@click.option("--out_dir", type=Path, required=True, help="cache root; art/ holds the shards")
@click.option("--generator", type=click.Choice(["audiox-maf", "tango-music-af-ft-mc"]),
              default=CacheRunConfig.generator)
@click.option("--corpus", default=CacheRunConfig.corpus,
              help="comma list of version ids (dataset/registry.py)")
@click.option("--max_files", type=int, default=CacheRunConfig.max_files,
              help="clips per dataset; 0 fetches every shard, which is terabytes")
@click.option("--skip_files", type=int, default=CacheRunConfig.skip_files,
              help="clips per dataset an earlier run already cached", hidden=True)
@click.option("--chars_min", type=int, default=CacheRunConfig.chars_min,
              help="shortest metadata text kept for prompts", hidden=True)
@click.option("--clip_seconds", type=float, default=CacheRunConfig.clip_seconds, hidden=True)
@click.option("--cond_policy", default=CacheRunConfig.cond_policy,
              help="timing pair per clip; a name or name:weight mix", hidden=True)
@click.option("--crop_policy", default=CacheRunConfig.crop_policy,
              help="crop start per clip; a name or name:weight mix", hidden=True)
@click.option("--crop_margin", type=float, default=CacheRunConfig.crop_margin,
              help="seconds kept clear of both source edges", hidden=True)
@click.option("--channel_policy", default=CacheRunConfig.channel_policy,
              help="stereo, first or mean; folds duplicate one channel", hidden=True)
@click.option("--volume_norm", is_flag=True, default=CacheRunConfig.volume_norm,
              help="SAT parity: frozen LUFS norm, flip, silence drop", hidden=True)
@click.option("--peak_norm", is_flag=True, default=CacheRunConfig.peak_norm,
              help="upstream tango parity: drop DC, scale peak to 0.5", hidden=True)
@click.option("--short_policy", default=CacheRunConfig.short_policy,
              help="repeat loops short clips; drop skips them", hidden=True)
@click.option("--prompt_policy", default=CacheRunConfig.prompt_policy,
              type=click.Choice([prompt_policy, VERBATIM_POLICY]),
              help="verbatim keeps caption text as written", hidden=True)
@click.option("--noise_levels", default=CacheRunConfig.noise_levels)
@click.option("--noise_schedule", default=CacheRunConfig.noise_schedule,
              type=click.Choice(["cosine", "scaled_linear"]),
              help="scaled_linear indexes the tango train step; tango-music only", hidden=True)
@click.option("--shard_clips", type=int, default=CacheRunConfig.shard_clips, hidden=True)
@click.option("--batch_clips", type=int, default=CacheRunConfig.batch_clips, hidden=True)
@click.option("--exec_batch_clips", type=int, default=CacheRunConfig.exec_batch_clips,
              help="per-gpu runtime batch; batch_clips stays the protocol", hidden=True)
@click.option("--max_shards", type=int, default=CacheRunConfig.max_shards,
              help="cap shard count, 0 keeps all", hidden=True)
@click.option("--num_steps", type=int, default=CacheRunConfig.num_steps,
              help="steps at t=1, scaled per level", hidden=True)
@click.option("--cfg_scale", type=float, default=CacheRunConfig.cfg_scale,
              help="1.0 turns guidance off", hidden=True)
@click.option("--store_dtype", default=CacheRunConfig.store_dtype, hidden=True)
@click.option("--determinism", is_flag=True, default=CacheRunConfig.determinism,
              help="pin kernels, costs throughput", hidden=True)
@click.option("--vae_dtype", default=CacheRunConfig.vae_dtype, hidden=True)
@click.option("--gen_dtype", default=CacheRunConfig.gen_dtype,
              help="denoiser and text encoder dtype", hidden=True)
@click.option("--compile_blocks", is_flag=True, default=CacheRunConfig.compile_blocks,
              help="compile each denoiser block", hidden=True)
@click.option("--compile_mode", default=CacheRunConfig.compile_mode,
              type=click.Choice(["auto", "off", "blocks", "model", "cudagraph"]),
              help="audiox DiT compile; auto = fastest", hidden=True)
@click.option("--seed", type=int, default=CacheRunConfig.seed)
@click.option("--device", default=CacheRunConfig.device)
def main(**option_dict):
    cfg = CacheRunConfig(**option_dict)
    draw_cfg = CacheDrawConfig()
    cache_run = LatentCacheRun.make_from_config(cfg, draw_cfg)
    cache_run.run()


if __name__ == "__main__":
    main()
