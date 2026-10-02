from __future__ import annotations

from importlib import import_module
from pathlib import Path

from audiogen_eval_protocols import AudioboxSpec
import torch


class AudioBoxScorer:
    def __init__(self, model, spec: AudioboxSpec):
        self.model = model
        self.spec = spec

    # Load the upstream predictor from its pinned checkpoint.
    @classmethod
    def make_from_ckpt(cls, ckpt_dir: Path, spec: AudioboxSpec):
        # audiobox_aesthetics is optional, so only this backbone imports it
        infer = import_module("audiobox_aesthetics.infer")
        ckpt_path = Path(ckpt_dir) / spec.checkpoint_name
        model = infer.initialize_predictor(ckpt=str(ckpt_path))
        scorer = cls(model, spec)
        return scorer

    # Mean of each axis over the population.
    @torch.no_grad()
    def get_scores(self, audio_iter, sample_rate: int):
        total_dict = dict.fromkeys(self.spec.axis_list, 0.0)
        count = 0
        for audio_item in audio_iter:
            clip = audio_item.detach().float().cpu()
            # upstream reads in-memory audio from the path key
            score_list = self.model.forward([{"path": clip, "sample_rate": sample_rate}])
            for axis in self.spec.axis_list:
                total_dict[axis] += float(score_list[0][axis])
            count += 1
        mean_dict = {}
        for axis, total in total_dict.items():
            mean_dict[axis] = total / count
        return mean_dict
