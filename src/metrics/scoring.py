from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from audiogen_eval_protocols import MetricSuite, PannSpec
import numpy as np
import torch
from tqdm import tqdm

from .audiobox_aesthetics import AudioBoxScorer
from .fad import FadDistance, PannEmbedder, FadEmbedder


@dataclass
class MetricScore:
    metric_dict: dict
    evidence_dict: dict


class MetricRowCache:
    # Stored rows for this key, or None before any pass.
    def load_rows(self, store, key: str):
        path = store.cache_dir / f"{key}.npy"
        row_array = None
        if path.is_file():
            row_array = np.load(path)
        return row_array

    # Store rows atomically so reruns in one directory reuse them.
    def write_rows(self, store, key: str, row_array: np.ndarray):
        path = store.cache_dir / f"{key}.npy"
        # np.save appends .npy, so the tmp name ends in it
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, row_array)
        tmp.replace(path)

    # Stored FAD rows and clip boundaries, or None.
    def load_fad_rows(self, store, key: str):
        path = store.cache_dir / f"{key}.npz"
        stored_dict = None
        if path.is_file():
            with np.load(path) as stored:
                stored_dict = {
                    "embeddings": np.asarray(stored["embeddings"]),
                    "counts": np.asarray(stored["counts"]),
                }
        return stored_dict

    # Store FAD rows and clip boundaries together.
    def write_fad_rows(self, store, key: str, row_array, count_array):
        path = store.cache_dir / f"{key}.npz"
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, embeddings=row_array, counts=count_array)
        tmp.replace(path)


class MetricBackbone(ABC):
    def __init__(self, row_cache: MetricRowCache):
        self.row_cache = row_cache

    # Stream one source's clips at their scored length.
    def get_scored_clips(self, store, source: str, sample_rate: int, channels: int,
                         metric: str = ""):
        clip_iter = store.get_clips(source, sample_rate, channels, metric)
        label = metric or "metric"
        for clip_item in tqdm(clip_iter, total=store.count, desc=f"{label} {source}"):
            yield clip_item

    # Load the model, fill every source's rows, free the model.
    @abstractmethod
    def make_rows(self, store, name_list, source_list):
        ...


class PannBackbone(MetricBackbone):
    def __init__(self, row_cache: MetricRowCache, ckpt_dir: Path, device: str,
                 spec: PannSpec, logit_name_list: tuple[str, ...]):
        super().__init__(row_cache)
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.spec = spec
        self.logit_name_list = logit_name_list

    # Report whether any selected metric reads logits.
    def check_logits(self, name_list):
        wants_logits = False
        for name in name_list:
            if name in self.logit_name_list:
                wants_logits = True
        return wants_logits

    # CNN14 embeddings for one source under the fd_pann window.
    def make_source_rows(self, embedder: PannEmbedder, store, source: str):
        key = f"embed_{source}_pann"
        row_array = self.row_cache.load_rows(store, key)
        if row_array is None:
            window = store.get_window("fd_pann")
            clip_iter = self.get_scored_clips(store, source, self.spec.sample_rate, 1, "fd_pann")
            row_array = embedder.get_embeddings(
                clip_iter, self.spec.sample_rate, window.remove_dc, window.pad_samples)
            self.row_cache.write_rows(store, key, row_array)
        return row_array

    # CNN14 logits for one source.
    def make_source_logits(self, embedder: PannEmbedder, store, source: str):
        key = f"logit_{source}"
        logit_array = self.row_cache.load_rows(store, key)
        if logit_array is None:
            clip_iter = self.get_scored_clips(store, source, self.spec.sample_rate, 1, "fd_pann")
            logit_array = embedder.get_logits(clip_iter, self.spec.sample_rate)
            self.row_cache.write_rows(store, key, logit_array)
        return logit_array

    # CNN14 embeddings, plus logits when asked.
    def make_rows(self, store, name_list, source_list):
        embedder = PannEmbedder.make_from_ckpt(self.ckpt_dir, self.device, self.spec)
        wants_logits = self.check_logits(name_list)
        row_dict = {}
        for source in source_list:
            row_dict[("pann", source)] = self.make_source_rows(embedder, store, source)
            if wants_logits:
                logit_array = self.make_source_logits(embedder, store, source)
                row_dict[("pann_logit", source)] = logit_array
        del embedder
        torch.cuda.empty_cache()
        return row_dict


class FadBackbone(MetricBackbone):
    def __init__(self, row_cache: MetricRowCache, ckpt_dir: Path, device: str, spec):
        super().__init__(row_cache)
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.spec = spec

    # FAD rows and counts for one source.
    def make_source_rows(self, embedder: FadEmbedder, store, source: str):
        key = f"embed_{source}_fad"
        stored_dict = self.row_cache.load_fad_rows(store, key)
        if stored_dict is None:
            window = store.get_window("fad")
            clip_iter = self.get_scored_clips(
                store, source, self.spec.sample_rate, 1, "fad")
            fad_rows = embedder.get_rows(clip_iter, self.spec.sample_rate, window.remove_dc)
            self.row_cache.write_fad_rows(
                store, key, fad_rows.row_array, fad_rows.count_array)
            stored_dict = {
                "embeddings": fad_rows.row_array,
                "counts": fad_rows.count_array,
            }
        stored_dict["counts"] = stored_dict["counts"].astype(np.int64, copy=False)
        return stored_dict

    # FAD embeddings and clip boundaries per source.
    def make_rows(self, store, name_list, source_list):
        embedder = FadEmbedder.make_from_ckpt(self.ckpt_dir, self.device, self.spec)
        row_dict = {}
        for source in source_list:
            stored_dict = self.make_source_rows(embedder, store, source)
            row_dict[("fad", source)] = stored_dict["embeddings"]
            row_dict[("fad_count", source)] = stored_dict["counts"]
        del embedder
        torch.cuda.empty_cache()
        return row_dict


class AudioBoxBackbone(MetricBackbone):
    def __init__(self, row_cache: MetricRowCache, ckpt_dir: Path, spec):
        super().__init__(row_cache)
        self.ckpt_dir = ckpt_dir
        self.spec = spec

    # Aesthetics axes per source, reference included.
    def make_rows(self, store, name_list, source_list):
        scorer = AudioBoxScorer.make_from_ckpt(self.ckpt_dir, self.spec)
        row_dict = {}
        for source in source_list:
            clip_iter = self.get_scored_clips(store, source, self.spec.sample_rate, 1)
            row_dict[("audiobox", source)] = scorer.get_scores(clip_iter, self.spec.sample_rate)
        del scorer
        torch.cuda.empty_cache()
        return row_dict


class MetricReducer(ABC):
    # One metric's table, per pair or per source.
    @abstractmethod
    def make_value(self, row_dict: dict, pair_list, source_list):
        ...


class MetricFrechet(MetricReducer):
    def __init__(self, distance: FadDistance, backbone: str):
        self.distance = distance
        self.backbone = backbone

    # Frechet distance per pair over one backbone's rows.
    def make_value(self, row_dict: dict, pair_list, source_list):
        value_dict = {}
        for ref_source, gen_source in pair_list:
            ref_array = row_dict[(self.backbone, ref_source)]
            gen_array = row_dict[(self.backbone, gen_source)]
            value_dict[f"{ref_source}_{gen_source}"] = self.distance.get_fad(ref_array, gen_array)
        return value_dict


class MetricDivergence(MetricReducer):
    def __init__(self, distance: FadDistance, kl_eps: float):
        self.distance = distance
        self.kl_eps = kl_eps

    # Paired CNN14 softmax KL per pair.
    def make_value(self, row_dict: dict, pair_list, source_list):
        value_dict = {}
        for ref_source, gen_source in pair_list:
            ref_array = row_dict[("pann_logit", ref_source)]
            gen_array = row_dict[("pann_logit", gen_source)]
            value_dict[f"{ref_source}_{gen_source}"] = self.distance.get_kl_softmax(
                ref_array, gen_array, self.kl_eps)
        return value_dict


class MetricInception(MetricReducer):
    def __init__(self, distance: FadDistance, spec: PannSpec):
        self.distance = distance
        self.spec = spec

    # Mean inception score per source, split count held stable.
    def make_value(self, row_dict: dict, pair_list, source_list):
        splits = self.spec.isc_splits
        for source in source_list:
            splits = min(splits, len(row_dict[("pann_logit", source)]))
        value_dict = {}
        for source in source_list:
            logit_array = row_dict[("pann_logit", source)]
            value_dict[source] = self.distance.get_inception_mean(
                logit_array, splits, self.spec.isc_seed)
        return value_dict


class MetricPerSource(MetricReducer):
    def __init__(self, backbone: str, axis: str):
        self.backbone = backbone
        self.axis = axis

    # One axis of each source's stored scores.
    def make_value(self, row_dict: dict, pair_list, source_list):
        value_dict = {}
        for source in source_list:
            score_dict = row_dict[(self.backbone, source)]
            value_dict[source] = score_dict[self.axis]
        return value_dict


class MetricScoring:
    def __init__(self, suite: MetricSuite, distance: FadDistance, row_cache: MetricRowCache,
                 ckpt_dir: Path, device: str):
        self.suite = suite
        self.distance = distance
        self.row_cache = row_cache
        self.ckpt_dir = ckpt_dir
        self.device = device

    # Build the scorer over one metric suite.
    @classmethod
    def make_from_suite(cls, suite: MetricSuite, ckpt_dir: Path, device: str):
        distance = FadDistance(suite.fd_eps)
        row_cache = MetricRowCache()
        scoring = cls(suite, distance, row_cache, ckpt_dir, device)
        return scoring

    # The row filler behind one backbone name.
    def make_backbone(self, backbone: str):
        suite = self.suite
        if backbone == "pann":
            filler = PannBackbone(self.row_cache, self.ckpt_dir, self.device, suite.pann,
                                  suite.pann_logit_list)
        elif backbone == "fad":
            filler = FadBackbone(self.row_cache, self.ckpt_dir, self.device, suite.fad)
        else:
            filler = AudioBoxBackbone(self.row_cache, self.ckpt_dir, suite.audiobox)
        return filler

    # The reducer behind one metric name.
    def make_reducer(self, name: str):
        if name == "fd_pann":
            reducer = MetricFrechet(self.distance, "pann")
        elif name == "fad":
            reducer = MetricFrechet(self.distance, "fad")
        elif name == "kl_softmax":
            reducer = MetricDivergence(self.distance, self.suite.pann.kl_eps)
        elif name == "is_mean":
            reducer = MetricInception(self.distance, self.suite.pann)
        else:
            reducer = MetricPerSource("audiobox", name[:2].upper())
        return reducer

    # Selected metrics grouped by the backbone they need.
    def get_backbone_groups(self, name_list):
        group_dict = {}
        for name in sorted(name_list):
            spec = self.suite.get_metric_spec(name)
            group_dict.setdefault(spec.backbone, []).append(name)
        return group_dict

    # Sources the pairs read, sorted.
    def get_source_list(self, pair_list):
        source_set = set()
        for pair_item in pair_list:
            source_set.update(pair_item)
        source_list = sorted(source_set)
        return source_list

    # FAD rows per source, kept as evidence.
    def make_evidence(self, row_dict: dict, source_list):
        evidence_dict = {}
        for source in source_list:
            if ("fad", source) in row_dict:
                evidence_dict[source] = {
                    "embeddings": row_dict[("fad", source)],
                    "counts": row_dict[("fad_count", source)],
                }
        return evidence_dict

    # Score the selected metrics, one backbone at a time.
    def get_metrics(self, store, name_list, pair_list):
        group_dict = self.get_backbone_groups(name_list)
        source_list = self.get_source_list(pair_list)
        row_dict = {}
        for backbone in sorted(group_dict):
            filler = self.make_backbone(backbone)
            backbone_rows = filler.make_rows(store, group_dict[backbone], source_list)
            row_dict.update(backbone_rows)
        metric_dict = {}
        for name in name_list:
            reducer = self.make_reducer(name)
            metric_dict[name] = reducer.make_value(row_dict, pair_list, source_list)
        evidence_dict = self.make_evidence(row_dict, source_list)
        score = MetricScore(metric_dict=metric_dict, evidence_dict=evidence_dict)
        return score
