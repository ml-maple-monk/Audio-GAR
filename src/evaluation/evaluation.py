from __future__ import annotations

import io
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from audiogen_eval_protocols import EvalSpec
import numpy as np
import pandas
import soundfile as sf
import torch
from safetensors.torch import save as save_tensors
from torchaudio.functional import resample
from tqdm import tqdm

from ..dataset.eval_loader import SampleEvalLoader
from ..metrics.fad import PannEmbedder
from ..metrics.scoring import MetricScoring
from ..models.generator import get_generator_spec, get_speed_defaults, load_generator
from ..models.tango2.vendor.tokenizer_arch import HIFIGAN_16K_64, AttrDict, Generator
from ..models.tokenizer import get_codec_traits, load_tokenizer
from .generation_cache import LatentCacheConfig, LatentCacheNaming, LatentCacheReader
from .registry import EvalProtocol, EvalRunConfig, MetricWindow


@dataclass
class EvalPopulation:
    sample_list: list
    subset: str | None


@dataclass
class DecoderState:
    tokenizer: object
    metadata: dict
    sample_rate: int
    frame_rate: float
    audio_channels: int
    codec_hop: int
    target_samples: int


@dataclass
class EvalDecodeParts:
    population: EvalPopulation
    window_dict: dict
    decoder: DecoderState
    scoring: MetricScoring
    audio_clip: AudioClip


class AudioClip:
    # Fold to the wanted channel count, mean or first.
    def get_folded_channels(self, clip: torch.Tensor, channels: int, downmix: str):
        folded = clip
        if clip.shape[0] != channels:
            if downmix == "first":
                folded = clip[:1].repeat(channels, 1)  # (C, S)
            else:
                mono = clip.mean(dim=0, keepdim=True)  # (1, S)
                folded = mono.repeat(channels, 1)
        return folded

    # One file as a (C, S) tensor and its rate.
    def load_wav(self, path):
        wav, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
        clip = torch.from_numpy(wav.T)  # (C, S)
        return clip, source_rate

    # Load one clip at a fixed rate, layout and length.
    def load_clip(self, path, sample_rate: int, channels: int, target_samples: int,
                  downmix: str):
        clip, source_rate = self.load_wav(path)
        if downmix:
            # one scored channel in, duplicated across the codec layout
            clip = self.get_folded_channels(clip, 1, downmix)
        # fold first; resampling per channel commutes and costs more
        clip = self.get_folded_channels(clip, channels, downmix or "mean")
        clip = resample(clip, source_rate, sample_rate)
        # reconstruction stacks clips into one batch, so lengths must agree
        if clip.shape[-1] < target_samples:
            clip = torch.nn.functional.pad(clip, (0, target_samples - clip.shape[-1]))
        clip = clip[:, :target_samples]  # (C, S)
        return clip

    # Load one scored clip, cropped and never padded.
    def load_metric_clip(self, path, sample_rate: int, channels: int, seconds: float,
                         downmix: str):
        clip, source_rate = self.load_wav(path)
        clip = self.get_folded_channels(clip, channels, downmix)
        clip = resample(clip, source_rate, sample_rate)
        clip = clip[:, :round(seconds * sample_rate)]  # (C, S)
        return clip

    # Level one cropped clip to a target peak.
    def get_peak_scaled(self, clip: torch.Tensor, dbfs: float):
        peak = clip.abs().max()
        scaled = clip
        if peak > 0:
            scaled = clip / peak * (10 ** (dbfs / 20))
        return scaled

    # Level one cropped clip to a target rms.
    def get_rms_scaled(self, clip: torch.Tensor, dbfs: float):
        rms = clip.pow(2).mean().sqrt()
        scaled = clip
        if rms > 0:
            scaled = (clip / rms * (10 ** (dbfs / 20))).clamp(-1.0, 1.0)
        return scaled

    # Tango and AudioLDM training level: no DC, half peak.
    def get_upstream_leveled(self, clip: torch.Tensor):
        centered = clip - clip.mean()
        leveled = centered / (centered.abs().max() + 1e-8) * 0.5
        return leveled


class AudioStore:
    def __init__(self, writer, sample_list: list, sample_rate: int, channels: int,
                 window_dict: dict, default_metric: str, level_policy: str,
                 audio_clip: AudioClip, gt_loader):
        self.writer = writer
        self.sample_list = sample_list
        self.sample_rate = sample_rate
        self.channels = channels
        self.window_dict = window_dict
        self.default_metric = default_metric
        self.level_policy = level_policy
        self.audio_clip = audio_clip
        self.gt_loader = gt_loader
        self.cache_dir = writer.root
        self.count = len(sample_list)
        self.torn_line_list = []

    # The pinned window this metric scores under.
    def get_window(self, metric: str = ""):
        window = self.window_dict[metric or self.default_metric]
        return window

    # Report whether this clip already has stored audio.
    def check_wav(self, source: str, sample):
        path = self.cache_dir / f"{source}_{sample.sample_id}.wav"
        stored = path.is_file() and path.stat().st_size > 0
        return stored

    # Store one synthesized clip and journal it.
    def write_wav(self, source: str, sample, wav: torch.Tensor):
        path = self.cache_dir / f"{source}_{sample.sample_id}.wav"
        # a kill must not leave a short wav
        tmp = path.with_suffix(".wav.tmp")
        if self.level_policy == "native":
            # reconstructions keep the reference level, unquantized
            float_array = wav.float().cpu().T.numpy()
            sf.write(tmp, float_array, self.sample_rate, format="WAV", subtype="FLOAT")
        else:
            # parity with the audio behind AudioX's reported numbers
            peak = wav.abs().max()
            pcm = wav.div(peak).clamp(-1, 1).float().cpu().T.numpy()
            sf.write(tmp, pcm, self.sample_rate, format="WAV", subtype="PCM_16")
        tmp.replace(path)
        self.writer.log({"source": source, "id": sample.sample_id, "ok": True})

    # Every progress row this output directory accumulated.
    def collect_journal(self):
        path = self.writer.get_progress_path()
        row_list = []
        if path.is_file():
            text = path.read_text()
            for line in text.splitlines():
                if line.strip():
                    try:
                        row_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        self.torn_line_list.append(line)
        if self.torn_line_list:
            # sharded appends interleave, and the journal feeds no metric
            print(f"journal dropped {len(self.torn_line_list)} rows torn by concurrent shards")
        return row_list

    # Captions in population order, empty when unlabeled.
    def get_prompts(self):
        prompt_list = []
        for sample_item in self.sample_list:
            prompt_list.append(str(sample_item.caption or ""))
        return prompt_list

    # Stream one source's clips under one metric's pinned policy.
    def get_clips(self, source: str, sample_rate: int, channels: int, metric: str = ""):
        for clip_idx in range(self.count):
            yield self.get_clip_at(clip_idx, source, sample_rate, channels, metric)

    # One source's raw clip at this population index.
    def load_source_clip(self, sample, source: str, sample_rate: int, channels: int,
                         window: MetricWindow):
        if source == "ref":
            clip = self.audio_clip.load_metric_clip(
                sample.audio_path, sample_rate, channels, window.seconds, window.downmix)
        elif source == "gt":
            gt_clip = self.gt_loader(sample)  # (C, S)
            clip = resample(gt_clip, self.sample_rate, sample_rate)
        else:
            path = self.cache_dir / f"{source}_{sample.sample_id}.wav"
            stored_clip, _ = self.audio_clip.load_wav(path)
            clip = resample(stored_clip, self.sample_rate, sample_rate)
        return clip

    # One source's clip at this population index, ready to embed.
    def get_clip_at(self, clip_idx: int, source: str, sample_rate: int, channels: int,
                    metric: str = ""):
        window = self.get_window(metric)
        sample = self.sample_list[clip_idx]
        clip = self.load_source_clip(sample, source, sample_rate, channels, window)
        clip = self.audio_clip.get_folded_channels(clip, channels, window.downmix)
        clip = clip[:, :round(window.seconds * sample_rate)]  # (C, S)
        leveled = window.renorm_gen
        if source == "ref":
            leveled = window.renorm_ref
        if leveled:
            clip = self.audio_clip.get_peak_scaled(clip, window.peak_dbfs)
        if window.rms_norm:
            clip = self.audio_clip.get_rms_scaled(clip, window.rms_dbfs)
        if window.band_rate:
            # hold both sources to the reference storage rate
            banded = resample(clip, sample_rate, window.band_rate)
            clip = resample(banded, window.band_rate, sample_rate)
        return clip


class DecoderCkptLoader:
    # Raw decoder tensors, compile wrapper names stripped.
    def get_raw_weights(self, state: dict):
        weight_dict = {}
        for key, value in state["decoder"].items():
            weight_dict[key.replace("_orig_mod.", "")] = value
        return weight_dict

    # EMA decoder tensors under the ema_model prefix.
    def get_ema_weights(self, state: dict):
        prefix = "ema_model."
        weight_dict = {}
        for key, value in state["ema"].items():
            if key.startswith(prefix):
                weight_dict[key[len(prefix):]] = value
        return weight_dict

    # Load finetuned decoder weights into the live codec.
    def load_decoder_weights(self, ckpt_path: Path, target, weights: str):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if weights == "ema":
            weight_dict = self.get_ema_weights(state)
        else:
            weight_dict = self.get_raw_weights(state)
        target.load_state_dict(weight_dict, strict=True)
        metadata = {
            "checkpoint_basename": ckpt_path.name,
            "checkpoint_step": state.get("step", 0),
            "decoder_weights": weights,
        }
        return metadata

    # Load a weight-normed HiFi-GAN bundle, fused, into the codec.
    def load_vocoder_weights(self, ckpt_path: Path, vae):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        generator = Generator(AttrDict(HIFIGAN_16K_64))
        generator.load_state_dict(state["generator"], strict=True)
        generator.remove_weight_norm()
        vae.vocoder.load_state_dict(generator.state_dict(), strict=True)
        metadata = {
            "vocoder_checkpoint_basename": ckpt_path.name,
            "vocoder_checkpoint_step": state.get("step", 0),
        }
        return metadata


class EvalPartLoader:
    def __init__(self, data_root: Path, eval_protocol: EvalProtocol):
        self.data_root = data_root
        self.eval_protocol = eval_protocol
        self.sample_loader = SampleEvalLoader(data_root)
        self.ckpt_loader = DecoderCkptLoader()

    # Every caption row in release order.
    def load_population(self, cfg: EvalRunConfig, spec: EvalSpec):
        subset = self.eval_protocol.eval_suite.get_subset(cfg.dataset, cfg.subset)
        sample_list = self.sample_loader.load_eval_dataset(cfg.dataset, subset)
        if cfg.num_clips:
            sample_list = sample_list[:cfg.num_clips]
        population = EvalPopulation(sample_list=sample_list, subset=subset)
        return population

    # Frames the scored window spans on the codec grid.
    def get_decoder_frames(self, model: str, seconds: float):
        spec = get_generator_spec(model)
        frames = round(seconds * spec.sample_rate / spec.downsampling_ratio)
        return frames

    # Mel bundle names over the live codec modules.
    def make_mel_decoder_target(self, vae):
        target = torch.nn.Module()
        target.post_quant_conv = vae.post_quant_conv
        target.decoder = vae.decoder
        return target

    # Finetuned decoder weights, or the pretrained labels.
    def load_decoder_metadata(self, cfg: EvalRunConfig, tokenizer):
        metadata = {
            "checkpoint_basename": "pretrained",
            "checkpoint_step": 0,
            "decoder_weights": "pretrained",
        }
        if cfg.decoder_ckpt:
            vae = tokenizer.get_vae_module()
            target = vae.decoder
            if get_codec_traits(cfg.model).decoder_target == "mel":
                target = self.make_mel_decoder_target(vae)
            metadata = self.ckpt_loader.load_decoder_weights(
                Path(cfg.decoder_ckpt), target, cfg.decoder_weights)
        return metadata

    # Open the pinned codec and fix the decode geometry.
    def load_decoder(self, cfg: EvalRunConfig, frames: int):
        tokenizer = load_tokenizer(cfg.model, ckpt_path=cfg.base_ckpt, device=cfg.device)
        metadata = self.load_decoder_metadata(cfg, tokenizer)
        if cfg.vocoder_ckpt:
            vae = tokenizer.get_vae_module()
            vocoder_metadata = self.ckpt_loader.load_vocoder_weights(Path(cfg.vocoder_ckpt), vae)
            metadata.update(vocoder_metadata)
        tokenizer.eval()
        codec_hop = round(tokenizer.sample_rate / tokenizer.frame_rate)
        decoder = DecoderState(
            tokenizer=tokenizer,
            metadata=metadata,
            sample_rate=tokenizer.sample_rate,
            frame_rate=tokenizer.frame_rate,
            audio_channels=tokenizer.audio_channels,
            codec_hop=codec_hop,
            target_samples=frames * codec_hop,
        )
        return decoder

    # Metric scorer over the protocol's metric suite.
    def make_scoring(self, cfg: EvalRunConfig):
        ckpt_dir = self.data_root / "fad_checkpoints"
        suite = self.eval_protocol.metric_suite
        scoring = MetricScoring.make_from_suite(suite, ckpt_dir, cfg.device)
        return scoring

    # Population, windows, codec and scorer a decode lane shares.
    def load_decode_parts(self, cfg: EvalRunConfig, spec: EvalSpec, frames: int):
        population = self.load_population(cfg, spec)
        window_dict = cfg.window_dict
        decoder = self.load_decoder(cfg, frames)
        scoring = self.make_scoring(cfg)
        parts = EvalDecodeParts(
            population=population, window_dict=window_dict, decoder=decoder,
            scoring=scoring, audio_clip=AudioClip(),
        )
        return parts


class EvalTask(ABC):
    def __init__(self, cfg: EvalRunConfig, spec: EvalSpec, eval_protocol: EvalProtocol,
                 population: EvalPopulation):
        self.cfg = cfg
        self.spec = spec
        self.eval_protocol = eval_protocol
        self.population = population
        self.exp_name = spec.exp_name
        self.protocol_name = eval_protocol.get_name()

    # Artifact config fields for this evaluation.
    @abstractmethod
    def get_config(self):
        ...

    # Produce this evaluation's artifacts and return the summary.
    @abstractmethod
    def run_evaluation(self, writer):
        ...


class EvalDecodeTask(EvalTask):
    def __init__(self, cfg: EvalRunConfig, spec: EvalSpec, eval_protocol: EvalProtocol,
                 parts: EvalDecodeParts, level_policy: str):
        super().__init__(cfg, spec, eval_protocol, parts.population)
        self.decoder = parts.decoder
        self.window_dict = parts.window_dict
        self.scoring = parts.scoring
        self.audio_clip = parts.audio_clip
        self.level_policy = level_policy

    # Config fields both decoder lanes share.
    def get_decoder_config(self):
        config = {
            "model": self.cfg.model,
            "dataset": self.cfg.dataset,
            "subset": self.population.subset,
            "clips": len(self.population.sample_list),
            "batch_size": self.cfg.batch_size,
            "metric_names": list(self.spec.metric_list),
            "protocol": self.protocol_name,
        }
        config.update(self.decoder.metadata)
        return config

    # Where the encoder input for one gt clip comes from.
    def load_gt_clip(self, sample):
        clip = None
        return clip

    # One store over this worker's population slice.
    def make_store(self, writer):
        sample_list = self.population.sample_list[self.cfg.shard::self.cfg.num_shards]
        store = AudioStore(
            writer, sample_list, self.decoder.sample_rate, self.decoder.audio_channels,
            self.window_dict, self.eval_protocol.eval_suite.default_metric, self.level_policy,
            self.audio_clip, self.load_gt_clip,
        )
        return store

    # Production units, each synthesized in one call.
    @abstractmethod
    def get_units(self, store: AudioStore):
        ...

    # Waveforms for one production unit.
    @abstractmethod
    def make_wavs(self, sample_list: list):
        ...

    # Write missing clips, one production unit at a time.
    def decode_audio(self, store: AudioStore):
        source = self.spec.source
        resumed = 0
        for unit_list in tqdm(self.get_units(store), desc=f"{source} {self.cfg.dataset}"):
            pending_list = []
            for sample_item in unit_list:
                if not store.check_wav(source, sample_item):
                    pending_list.append(sample_item)
            resumed += len(unit_list) - len(pending_list)
            if pending_list:
                wav_list = self.make_wavs(pending_list)
                for sample_item, wav in zip(pending_list, wav_list, strict=True):
                    store.write_wav(source, sample_item, wav)
        resumed_dict = {source: resumed}
        return resumed_dict

    # Synthesize this worker's clips and journal them.
    def make_audio(self, writer, store: AudioStore):
        run_dir_resumed = any(writer.root.glob("*.wav"))
        decoded_dict = self.decode_audio(store)
        resumed_dict = {"run_dir": run_dir_resumed}
        resumed_dict.update(decoded_dict)
        self.decoder.tokenizer = None
        torch.cuda.empty_cache()
        journal = store.collect_journal()
        writer.write("fad_progress", journal)
        return resumed_dict

    # Pack ordered clips with their exact FAD row boundaries.
    def make_fad_row_bundle(self, sample_list: list, evidence_dict: dict):
        id_list = []
        for sample_item in sample_list:
            id_list.append(sample_item.sample_id)
        payload = {"sample_ids": np.asarray(id_list)}
        for source, rows in evidence_dict.items():
            payload[f"embed_{source}_fad"] = rows["embeddings"]
            payload[f"{source}_counts"] = rows["counts"]
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **payload)
        bundle = buffer.getvalue()
        return bundle

    # Flat metrics plus the pinned summary fields.
    def make_summary(self, metric_dict: dict, config: dict):
        summary_dict = {}
        for key, row in metric_dict.items():
            for name, value in row.items():
                summary_dict[f"{key}_{name}"] = value
        for name in self.spec.summary_field_list:
            summary_dict[name] = config[name]
        return summary_dict

    # Score the stored audio and write the evaluation artifact.
    def write_scores(self, writer, store: AudioStore, resumed_dict: dict):
        score = self.scoring.get_metrics(store, self.spec.metric_list, self.spec.pair_list)
        if "fad" in self.spec.metric_list:
            bundle = self.make_fad_row_bundle(store.sample_list, score.evidence_dict)
            writer.write("fad_rows", bundle)
        config = self.get_config()
        artifact = dict(config)
        artifact.update({
            "resumed": resumed_dict, "metrics": score.metric_dict,
        })
        writer.write(self.exp_name, artifact)
        summary_dict = self.make_summary(score.metric_dict, config)
        return summary_dict

    # Synthesize, score, write evidence and return the summary.
    def run_evaluation(self, writer):
        store = self.make_store(writer)
        resumed_dict = self.make_audio(writer, store)
        if self.cfg.num_shards > 1:
            summary_dict = {
                "clips": len(self.population.sample_list), "shard": self.cfg.shard,
                "num_shards": self.cfg.num_shards,
            }
        else:
            summary_dict = self.write_scores(writer, store, resumed_dict)
        return summary_dict


class EvalReconRefTask(EvalDecodeTask):
    def __init__(self, cfg: EvalRunConfig, spec: EvalSpec, eval_protocol: EvalProtocol,
                 parts: EvalDecodeParts, frames: int):
        super().__init__(cfg, spec, eval_protocol, parts, "native")
        self.frames = frames
        self.clip_index = {}

    # Build the reconstruction lane from a resolved config.
    @classmethod
    def make_from_config(cls, cfg: EvalRunConfig, spec: EvalSpec, eval_protocol: EvalProtocol,
                         loader: EvalPartLoader):
        frames = cfg.decoder_frames or loader.get_decoder_frames(cfg.model, cfg.seconds)
        parts = loader.load_decode_parts(cfg, spec, frames)
        task = cls(cfg, spec, eval_protocol, parts, frames)
        return task

    # Artifact config fields for reconstruction.
    def get_config(self):
        config = self.get_decoder_config()
        config.update({
            "frames": self.frames,
            "samples": self.decoder.target_samples,
            "recon_level": self.cfg.recon_level,
        })
        return config

    # Pad true length to codec grid, capped at the window.
    def get_gt_target(self, true_samples: int):
        hop = self.decoder.codec_hop
        padded = -(-true_samples // hop) * hop
        target = min(self.decoder.target_samples, padded)
        return target

    # The clip the encoder saw, at storage rate.
    def load_gt_clip(self, sample):
        info = sf.info(str(sample.audio_path))
        true_samples = round(info.frames * self.decoder.sample_rate / info.samplerate)
        target = self.get_gt_target(true_samples)
        clip = self.audio_clip.load_clip(
            sample.audio_path, self.decoder.sample_rate, self.decoder.audio_channels,
            target, self.spec.gt_downmix,
        )
        return clip

    # One clip per unit; true lengths differ per file.
    def get_units(self, store: AudioStore):
        unit_list = []
        for clip_idx, sample_item in enumerate(store.sample_list):
            self.clip_index[sample_item.sample_id] = clip_idx
            unit_list.append([sample_item])
        return unit_list

    # Reconstruct one clip through the pinned decoder.
    def make_wavs(self, sample_list: list):
        sample = sample_list[0]
        clip = self.load_gt_clip(sample)  # (C, S)
        if self.cfg.recon_level == "upstream":
            clip = self.audio_clip.get_upstream_leveled(clip)
        tokenizer = self.decoder.tokenizer
        batch = clip[None].to(tokenizer.device)  # (1, C, S)
        if self.cfg.recon_eps == "fresh":
            wav_batch = tokenizer.reconstruct(batch)  # (1, C, S)
        else:
            clip_seed = self.cfg.seed + self.cfg.recon_seed_offset
            clip_seed += self.clip_index[sample.sample_id]
            torch.manual_seed(clip_seed)
            wav_batch = tokenizer.reconstruct(batch)  # (1, C, S)
        wav_list = [wav_batch[0]]
        return wav_list


class EvalGenRefTask(EvalDecodeTask):
    def __init__(self, cfg: EvalRunConfig, spec: EvalSpec, eval_protocol: EvalProtocol,
                 parts: EvalDecodeParts, sampling: dict, latent_set):
        super().__init__(cfg, spec, eval_protocol, parts, cfg.arm.level_policy)
        self.sampling = sampling
        self.latent_set = latent_set

    # The pinned sampling fields this artifact records.
    @classmethod
    def make_sampling(cls, cfg: EvalRunConfig, spec: EvalSpec, eval_protocol: EvalProtocol,
                      frames: int):
        naming = LatentCacheNaming(LatentCacheConfig())
        cache_sampling = naming.make_sampling(
            cfg.model, frames, cfg.seconds, cfg.arm, cfg.cfg_scale, eval_protocol.get_name())
        sampling = {}
        for name in spec.sampling_field_list:
            sampling[name] = cache_sampling[name]
        return sampling

    # Build the generation lane from a resolved config.
    @classmethod
    def make_from_config(cls, cfg: EvalRunConfig, spec: EvalSpec, eval_protocol: EvalProtocol,
                         loader: EvalPartLoader):
        frames = cfg.decoder_frames or get_generator_spec(cfg.model).generation_frames
        parts = loader.load_decode_parts(cfg, spec, frames)
        sampling = cls.make_sampling(cfg, spec, eval_protocol, frames)
        reader = LatentCacheReader(LatentCacheConfig(artifact_dir=cfg.cache_artifact_dir))
        latent_set = reader.load_cache(Path(cfg.generation_cache))
        task = cls(cfg, spec, eval_protocol, parts, sampling, latent_set)
        return task

    # Artifact config fields for generation.
    def get_config(self):
        config = self.get_decoder_config()
        config["seconds"] = self.cfg.seconds
        config.update(self.sampling)
        return config

    # Fixed-size batches; cached latents share a shape.
    def get_units(self, store: AudioStore):
        size = self.cfg.batch_size
        unit_list = []
        for start in range(0, len(store.sample_list), size):
            unit_list.append(store.sample_list[start:start + size])
        return unit_list

    # Decode one batch of cached latents to waveforms.
    @torch.no_grad()
    def make_wavs(self, sample_list: list):
        tokenizer = self.decoder.tokenizer
        latent = self.latent_set.load_samples(sample_list)  # (B, D, T)
        latent = latent.to(tokenizer.device)
        denormalized = tokenizer.apply_latent_scale(latent, "denormalize")  # (B, D, T)
        wav_batch = tokenizer.decode(denormalized)
        wav_list = list(wav_batch)
        return wav_list


class EvalGenRefCacheTask(EvalTask):
    def __init__(self, cfg: EvalRunConfig, spec: EvalSpec, eval_protocol: EvalProtocol,
                 population: EvalPopulation, generator, naming: LatentCacheNaming,
                 header: dict, frames: int):
        super().__init__(cfg, spec, eval_protocol, population)
        self.generator = generator
        self.naming = naming
        self.header = header
        self.frames = frames

    # Family cache switches; families without defaults load unchanged.
    @classmethod
    def get_speed_switches(cls, cfg: EvalRunConfig):
        defaults = get_speed_defaults(cfg.model)
        switch_dict = {}
        if defaults is not None:
            switch_dict["gen_dtype"] = cfg.gen_dtype or defaults["gen_dtype"]
        return switch_dict

    # The published cache header for this population.
    @classmethod
    def make_header(cls, cfg: EvalRunConfig, population: EvalPopulation, sampling: dict):
        clip_count = len(population.sample_list)
        header = {
            "generator": cfg.model,
            "dataset": cfg.dataset,
            "subset": population.subset,
        }
        header.update(sampling)
        header.update({
            "clips": clip_count,
            "shards": (clip_count + cfg.cache_shard_clips - 1) // cfg.cache_shard_clips,
            "shard_clips": cfg.cache_shard_clips,
            "dtype": cfg.cache_dtype,
            "seed": cfg.seed,
        })
        return header

    # Build the latent cache lane from a resolved config.
    @classmethod
    def make_from_config(cls, cfg: EvalRunConfig, spec: EvalSpec, eval_protocol: EvalProtocol,
                         loader: EvalPartLoader):
        population = loader.load_population(cfg, spec)
        frames = cfg.decoder_frames or get_generator_spec(cfg.model).generation_frames
        cache_cfg = LatentCacheConfig()
        naming = LatentCacheNaming(cache_cfg)
        sampling = naming.make_sampling(
            cfg.model, frames, cfg.seconds, cfg.arm, cfg.cfg_scale, eval_protocol.get_name())
        header = cls.make_header(cfg, population, sampling)
        switch_dict = cls.get_speed_switches(cfg)
        generator = load_generator(
            cfg.model, ckpt_path=cfg.base_ckpt, device=cfg.device, **switch_dict)
        generator.apply_sampling(cfg.arm.steps, sampling["cfg_scale"])
        task = cls(cfg, spec, eval_protocol, population, generator, naming, header, frames)
        return task

    # Artifact config fields for the cache.
    def get_config(self):
        config = dict(self.header)
        config["batch_size"] = self.cfg.batch_size
        switch_dict = self.get_speed_switches(self.cfg)
        config.update(switch_dict)
        return config

    # Shards this worker owns under the round-robin split.
    def get_owned_shards(self):
        owned_list = []
        for shard in range(self.header["shards"]):
            if shard % self.cfg.num_shards == self.cfg.shard:
                owned_list.append(shard)
        return owned_list

    # Report whether both artifacts of one shard exist.
    def check_shard_done(self, writer, shard: int):
        item = self.naming.make_shard_item(shard)
        done = writer.done("gen_ref_latent_shard", item)
        done = done and writer.done("gen_ref_latent_clip", item)
        return done

    # This worker's shards still missing an artifact.
    def get_pending_shards(self, writer):
        pending_list = []
        for shard in self.get_owned_shards():
            if not self.check_shard_done(writer, shard):
                pending_list.append(shard)
        return pending_list

    # Generate at native length, store the leading frames.
    def generate_latent_shard(self, row_list: list):
        arm = self.cfg.arm
        native = self.generator.spec.generation_frames
        dtype = self.naming.get_torch_dtype(self.cfg.cache_dtype)
        part_list = []
        for start in range(0, len(row_list), self.cfg.batch_size):
            batch = row_list[start:start + self.cfg.batch_size]
            prompt_list = []
            seed_list = []
            for row_item in batch:
                prompt_list.append(row_item["prompt"])
                seed_list.append(row_item["seed"])
            # the sampler owns the window; storage keeps its head
            latent = self.generator.generate_latent(
                prompt_list, arm.steps, arm.seconds_start, arm.seconds_total, seed_list, native,
            )  # (B, D, T)
            head = latent[..., :self.frames]
            part_list.append(head.to(dtype).cpu())
        shard_latent = torch.cat(part_list)  # (N, D, F)
        return shard_latent

    # Report whether every shard of the cache exists.
    def check_complete(self, writer):
        complete = True
        for shard in range(self.header["shards"]):
            complete = complete and self.check_shard_done(writer, shard)
        return complete

    # Publish the header only after every shard exists.
    def write_cache_header(self, writer):
        complete = self.check_complete(writer)
        if complete and not writer.done("gen_ref_latent_cache"):
            writer.write("gen_ref_latent_cache", self.header)
        return complete

    # Generate this worker's missing cache shards.
    def run_evaluation(self, writer):
        pending_list = self.get_pending_shards(writer)
        size = self.cfg.cache_shard_clips
        for shard in tqdm(pending_list, unit="shard"):
            start = shard * size
            sample_list = self.population.sample_list[start:start + size]
            row_list = self.naming.make_rows(self.cfg.dataset, shard, start, sample_list,
                                             self.cfg.seed)
            latent = self.generate_latent_shard(row_list)  # (N, D, F)
            item = self.naming.make_shard_item(shard)
            writer.write("gen_ref_latent_shard", save_tensors({"z": latent}), item=item)
            writer.write("gen_ref_latent_clip", row_list, item=item)
            writer.log({"source": "latent", "id": item, "ok": True})
        complete = self.write_cache_header(writer)
        del self.generator
        torch.cuda.empty_cache()
        mine = len(self.get_owned_shards())
        summary_dict = {
            "dataset": self.cfg.dataset, "clips": len(self.population.sample_list),
            "shard": self.cfg.shard, "num_shards": self.cfg.num_shards,
            "cache_shards": mine, "steps": self.header["steps"], "complete": complete,
        }
        return summary_dict


class EvalGenRefEmbedTask(EvalTask):
    def __init__(self, gen_ref_task: EvalGenRefTask):
        super().__init__(gen_ref_task.cfg, gen_ref_task.spec, gen_ref_task.eval_protocol,
                         gen_ref_task.population)
        self.gen_ref_task = gen_ref_task

    # Wrap a generation lane over the same cache.
    @classmethod
    def make_from_config(cls, cfg: EvalRunConfig, spec: EvalSpec, eval_protocol: EvalProtocol,
                         loader: EvalPartLoader):
        gen_ref_task = EvalGenRefTask.make_from_config(cfg, spec, eval_protocol, loader)
        task = cls(gen_ref_task)
        return task

    # Artifact config fields, shared with the generation lane.
    def get_config(self):
        config = self.gen_ref_task.get_config()
        return config

    # AudioSet index, mid and label columns.
    def get_class_table(self):
        path = Path(__file__).parents[1] / "metrics" / "panns_data" / "class_labels_indices.csv"
        class_df = pandas.read_csv(path, dtype=str, keep_default_na=False)
        index_array = class_df["index"].astype(int).to_numpy()
        table = {
            "class_idx": index_array.astype(np.int16),
            "class_mids": np.array(class_df["mid"].tolist()),
            "class_labels": np.array(class_df["display_name"].tolist()),
        }
        return table

    # Yield scored clips, recording their sample ids.
    def get_labeled_clips(self, store: AudioStore, source: str, rate: int, id_list: list):
        clip_iter = store.get_clips(source, rate, 1, "fd_pann")
        for sample_item, clip in zip(store.sample_list, clip_iter, strict=True):
            id_list.append(sample_item.sample_id)
            yield clip

    # Reference rows reused from a donor artifact.
    def load_ref_rows(self, path: Path, id_by_source: dict):
        name_list = ["embed_ref_pann", "logit_ref"]
        array_dict = {}
        with np.load(path) as stored:
            id_list = []
            for value in stored["sample_ids"]:
                id_list.append(str(value))
            for name in name_list:
                array_dict[name] = stored[name]
        id_by_source["pann_ref"] = id_list
        return array_dict

    # Serialize the ordered row artifact.
    def make_embed_rows(self, sample_id_list: list, array_dict: dict):
        payload = {"sample_ids": np.array(sample_id_list)}
        payload.update(array_dict)
        class_table = self.get_class_table()
        payload.update(class_table)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **payload)
        rows = buffer.getvalue()
        return rows

    # Compiled bf16 CNN14 rows and logits per source.
    def predict_fd_pann(self, store: AudioStore, id_by_source: dict, source_list: tuple):
        suite = self.eval_protocol.metric_suite
        ckpt_dir = self.gen_ref_task.scoring.ckpt_dir
        loaded = PannEmbedder.make_from_ckpt(ckpt_dir, self.cfg.device, suite.pann)
        compiled_model = torch.compile(loaded.model)
        embedder = PannEmbedder(compiled_model, suite.pann, self.cfg.device)
        window = store.get_window("fd_pann")
        array_dict = {}
        for source in source_list:
            id_list = []
            clip_iter = self.get_labeled_clips(store, source, suite.pann.sample_rate, id_list)
            result = embedder.get_rows_logits(
                clip_iter, suite.pann.sample_rate, window.remove_dc, window.pad_samples)
            id_by_source[f"pann_{source}"] = id_list
            array_dict[f"embed_{source}_pann"] = result.row_array
            array_dict[f"logit_{source}"] = result.logit_array
        del embedder
        del loaded
        torch.cuda.empty_cache()
        return array_dict

    # Every row array, reference rows loaded or predicted.
    def make_row_arrays(self, store: AudioStore, resumed_dict: dict):
        id_by_source = {}
        ref_array_dict = {}
        source_list = ("ref", "gen")
        resumed_dict["ref_rows"] = "predicted"
        if self.cfg.ref_rows:
            ref_array_dict = self.load_ref_rows(Path(self.cfg.ref_rows), id_by_source)
            source_list = ("gen",)
            resumed_dict["ref_rows"] = "loaded"
        array_dict = self.predict_fd_pann(store, id_by_source, source_list)
        array_dict.update(ref_array_dict)
        return array_dict

    # Decode fresh, embed both sources, publish the row artifact.
    def run_evaluation(self, writer):
        gen_ref_task = self.gen_ref_task
        store = gen_ref_task.make_store(writer)
        device_type = torch.device(self.cfg.device).type
        with torch.autocast(device_type, dtype=torch.bfloat16):
            resumed_dict = gen_ref_task.make_audio(writer, store)
        array_dict = self.make_row_arrays(store, resumed_dict)
        sample_id_list = []
        for sample_item in store.sample_list:
            sample_id_list.append(sample_item.sample_id)
        writer.write("gen_ref_embed_rows", self.make_embed_rows(sample_id_list, array_dict))
        distance = gen_ref_task.scoring.distance.get_fad(
            array_dict["embed_ref_pann"], array_dict["embed_gen_pann"])
        metric_dict = {"fd_pann": {"ref_gen": distance}}
        summary = self.get_config()
        summary["resumed"] = resumed_dict
        summary["metrics"] = metric_dict
        writer.write(self.exp_name, summary)
        summary_dict = gen_ref_task.make_summary(metric_dict, summary)
        return summary_dict


class EvalTaskMaker:
    def __init__(self, loader: EvalPartLoader):
        self.loader = loader

    # Build the evaluation lane the config names.
    def make_task(self, cfg: EvalRunConfig, spec: EvalSpec, eval_protocol: EvalProtocol):
        if cfg.evaluation == "recon_ref":
            lane_cls = EvalReconRefTask
        elif cfg.evaluation == "gen_ref":
            lane_cls = EvalGenRefTask
        elif cfg.evaluation == "gen_ref_cache":
            lane_cls = EvalGenRefCacheTask
        else:
            lane_cls = EvalGenRefEmbedTask
        task = lane_cls.make_from_config(cfg, spec, eval_protocol, self.loader)
        return task
