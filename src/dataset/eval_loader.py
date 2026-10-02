from __future__ import annotations

from pathlib import Path

import pandas

from .registry import SampleEval


class SampleEvalLoader:
    def __init__(self, data_root: Path):
        self.data_root = data_root

    # One manifest row as an evaluation record.
    def make_sample(self, name: str, root: Path, row: dict):
        audio_path = None
        if row["audio_file"]:
            audio_path = root / row["audio_file"]
        sample_rate = None
        if row.get("sample_rate"):
            sample_rate = int(row["sample_rate"])
        sample = SampleEval(
            dataset=name,
            sample_id=row["sample_id"],
            recording_id=row["recording_id"],
            caption=row["caption"],
            audio_path=audio_path,
            start_seconds=float(row["start_seconds"]),
            duration_seconds=float(row["duration_seconds"]),
            subset=row["subset"],
            sample_rate=sample_rate,
        )
        return sample

    # Records from the cached manifest, every split when unset.
    def load_eval_dataset(self, name: str, subset: str | None):
        root = self.data_root / "eval_datasets" / name
        # cells stay strings, so empty means absent
        manifest_df = pandas.read_csv(root / "manifest.csv", dtype=str, keep_default_na=False)
        if subset is not None:
            subset_mask = manifest_df["subset"] == subset
            manifest_df = manifest_df[subset_mask]
        record_list = manifest_df.to_dict("records")
        sample_list = []
        for record_item in record_list:
            sample = self.make_sample(name, root, record_item)
            sample_list.append(sample)
        return sample_list
