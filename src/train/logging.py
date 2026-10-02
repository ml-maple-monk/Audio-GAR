from __future__ import annotations

import sqlite3
from pathlib import Path

import mlflow

from ..models.stable_audio_open.registry import SpecTrainTracking


class TrainLogger:
    def __init__(self, db_path: Path, spec: SpecTrainTracking, run_name: str):
        self.spec = spec
        self.db_path = Path(db_path)
        tracking_uri = f"sqlite:///{self.db_path}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("decoder_finetune")
        mlflow.start_run(run_name=run_name)

    # Flatten nested run config into dotted text pairs.
    def make_param_rows(self, config: dict, prefix: str = ""):
        pair_list = []
        key_list = sorted(config)
        for key_item in key_list:
            value = config[key_item]
            name = f"{prefix}{key_item}"
            if isinstance(value, dict):
                child_list = self.make_param_rows(value, f"{name}.")
                pair_list.extend(child_list)
            else:
                pair_list.append((name, str(value)))
        return pair_list

    # Send the flattened run config to mlflow.
    def log_params(self, config: dict):
        pair_list = self.make_param_rows(config)
        param_dict = dict(pair_list)
        mlflow.log_params(param_dict)

    # Send one prefixed metric table at a step.
    def write_metrics(self, step: int, prefix: str, values: dict, elapsed: float):
        named_dict = {}
        for key_item, value_item in values.items():
            named_dict[f"{prefix}/{key_item}"] = float(value_item)
        named_dict[f"{prefix}/elapsed_seconds"] = float(elapsed)
        mlflow.log_metrics(named_dict, step=step)

    # Log the numeric fields of one optimizer update.
    def log_step(self, row: dict):
        value_dict = {}
        for key_item, value_item in row.items():
            numeric = isinstance(value_item, (int, float))
            if numeric and key_item not in ("index", "elapsed_seconds"):
                value_dict[key_item] = value_item
        prefix = f"train/{row['lane']}"
        self.write_metrics(row["index"], prefix, value_dict, row["elapsed_seconds"])

    # Log every held-out domain table at one step.
    def log_eval(self, step: int, elapsed: float, tables: dict):
        for domain_item, value_dict in tables.items():
            self.write_metrics(step, f"test/{domain_item}", value_dict, elapsed)

    # Close the mlflow run.
    def finish(self):
        mlflow.end_run()

    # Consistent full copy of the live backend store.
    def make_snapshot(self):
        # Connection.serialize needs 3.11; the image runs 3.10
        mirror_path = self.db_path.with_suffix(".snapshot")
        live = sqlite3.connect(self.db_path)
        mirror = sqlite3.connect(mirror_path)
        with mirror:
            live.backup(mirror)
        live.close()
        mirror.close()
        snapshot = mirror_path.read_bytes()
        return snapshot
