from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from ..models.generator import get_generator_spec


@dataclass(frozen=True)
class LatentCacheConfig:
    artifact_dir: str = "artifact"
    header_pattern: str = "gen_ref_latent_cache.json"
    clip_pattern: str = "gen_ref_latent_clip.*.jsonl"
    shard_pattern: str = "gen_ref_latent_shard.*.safetensors"
    shard_prefix: str = "gen_ref_latent_shard"
    legacy_ddim_eta: float = 0.0
    legacy_candidates: int = 1


@dataclass(frozen=True)
class LatentGeneration:
    index: int
    shard: int
    row: int
    sample_id: str
    recording_id: str
    prompt: str
    seed: int


class LatentCacheNaming:
    def __init__(self, cfg: LatentCacheConfig):
        self.cfg = cfg

    # Diffusion stream for one clip, offset by its population index.
    def make_clip_seed(self, index: int, seed: int):
        clip_seed = seed + index
        return clip_seed

    # Name one artifact by its shard index.
    def make_shard_item(self, shard: int):
        item = f"shard{shard:06d}"
        return item

    # Torch dtype behind one cache dtype name.
    def get_torch_dtype(self, dtype_name: str):
        dtype = getattr(torch, dtype_name)
        return dtype

    # Resolve the protocol sampling, geometry and guidance.
    def make_sampling(self, model: str, frames: int, seconds: float, arm, cfg_scale: float,
                      protocol_name: str):
        model_spec = get_generator_spec(model)
        sampling = model_spec.sampling
        cfg = cfg_scale or arm.cfg_scale
        cfg_source = "arm"
        if cfg != arm.cfg_scale:
            cfg_source = "override"
        sampling_dict = {
            "sample_rate": model_spec.sample_rate,
            "frame_rate": model_spec.sample_rate / model_spec.downsampling_ratio,
            "latent_dim": model_spec.latent_dim,
            "frames": frames,
            "samples": frames * model_spec.downsampling_ratio,
            "clip_seconds": seconds,
            "prompt_source": "dataset_caption",
            "steps": arm.steps,
            "steps_mode": "fixed",
            "sampler_type": sampling.sampler_type,
            "sigma_min": sampling.sigma_min,
            "sigma_max": sampling.sigma_max,
            "rho": sampling.rho,
            "cfg_scale": cfg,
            "cfg_source": cfg_source,
            "apg_scale": getattr(sampling, "apg_scale", 0.0),
            "ddim_eta": getattr(sampling, "ddim_eta", 0.0),
            "candidates": getattr(sampling, "candidates", 1),
            "batch_cfg": sampling.batch_cfg,
            "rescale_cfg": sampling.rescale_cfg,
            "seconds_start": arm.seconds_start,
            "seconds_total": arm.seconds_total,
            "protocol": protocol_name,
        }
        return sampling_dict

    # Record each cached prompt and stream for one shard.
    def make_rows(self, dataset: str, shard: int, start: int, sample_list: list, seed: int):
        row_list = []
        for row_idx, sample_item in enumerate(sample_list):
            row_list.append({
                "dataset": dataset,
                "index": start + row_idx,
                "sample_id": sample_item.sample_id,
                "recording_id": sample_item.recording_id,
                "prompt": str(sample_item.caption or ""),
                "shard": shard,
                "row": row_idx,
                "seed": self.make_clip_seed(start + row_idx, seed),
            })
        return row_list


class LatentCacheReader:
    def __init__(self, cfg: LatentCacheConfig):
        self.cfg = cfg

    # Cache artifacts matching one pattern, sorted by name.
    def get_cache_paths(self, cache_dir: Path, pattern: str):
        artifact_dir = cache_dir / self.cfg.artifact_dir
        path_list = sorted(artifact_dir.glob(pattern))
        return path_list

    # Read one cache header, filling legacy fields.
    def load_header(self, cache_dir: Path):
        path_list = self.get_cache_paths(cache_dir, self.cfg.header_pattern)
        text = path_list[0].read_text()
        header = json.loads(text)
        header.setdefault("ddim_eta", self.cfg.legacy_ddim_eta)
        header.setdefault("candidates", self.cfg.legacy_candidates)
        return header

    # Read the ordered manifest rows.
    def load_rows(self, cache_dir: Path):
        row_by_index = {}
        for path in self.get_cache_paths(cache_dir, self.cfg.clip_pattern):
            text = path.read_text()
            for line in text.splitlines():
                if line:
                    value = json.loads(line)
                    row_by_index[value["index"]] = value
        row_list = []
        for index in sorted(row_by_index):
            value = row_by_index[index]
            row_list.append(LatentGeneration(
                index=value["index"], shard=value["shard"], row=value["row"],
                sample_id=value["sample_id"], recording_id=value["recording_id"],
                prompt=value["prompt"], seed=value["seed"],
            ))
        return row_list

    # Open a finished cache for latent lookup.
    def load_cache(self, cache_dir: Path):
        header = self.load_header(cache_dir)
        row_list = self.load_rows(cache_dir)
        latent_set = LatentGenerationSet(self.cfg, cache_dir, header, row_list)
        return latent_set


class LatentGenerationSet:
    def __init__(self, cfg: LatentCacheConfig, cache_dir: Path, header: dict, row_list: list):
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.header = header
        self.row_list = row_list
        self.row_by_sample = {}
        for row_item in row_list:
            self.row_by_sample[row_item.sample_id] = row_item
        self.blob_dict = {}
        self.naming = LatentCacheNaming(cfg)

    # Load one latent shard, cached after the first read.
    def load_shard(self, shard: int):
        if shard not in self.blob_dict:
            item = self.naming.make_shard_item(shard)
            name = f"{self.cfg.shard_prefix}.{item}.safetensors"
            path = self.cache_dir / self.cfg.artifact_dir / name
            tensor_dict = load_file(path)
            self.blob_dict[shard] = tensor_dict["z"]  # (N, D, T)
        latent = self.blob_dict[shard]
        return latent

    # Load latents in requested sample order.
    def load_samples(self, sample_list: list):
        value_list = []
        for sample_item in sample_list:
            row_item = self.row_by_sample[sample_item.sample_id]
            shard = self.load_shard(row_item.shard)  # (N, D, T)
            value_list.append(shard[row_item.row].float())
        latent = torch.stack(value_list)  # (B, D, T)
        return latent
