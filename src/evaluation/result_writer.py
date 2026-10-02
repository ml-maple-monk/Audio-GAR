from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalWriterConfig:
    artifact_dir: str = "artifact"
    default_ext: str = ".json"
    fad_progress: str = ".jsonl"
    fad_rows: str = ".bin"
    gen_ref_embed_rows: str = ".bin"
    gen_ref_latent_shard: str = ".safetensors"
    gen_ref_latent_clip: str = ".jsonl"
    progress_name: str = "progress.jsonl"
    summary_name: str = "summary.json"


class EvalResultWriter:
    def __init__(self, cfg: EvalWriterConfig, root: Path):
        self.cfg = cfg
        self.root = root
        self.artifact_dir = root / cfg.artifact_dir

    # Open an output directory with its artifact folder.
    @classmethod
    def make_from_dir(cls, cfg: EvalWriterConfig, out_dir: Path):
        writer = cls(cfg, Path(out_dir))
        writer.artifact_dir.mkdir(parents=True, exist_ok=True)
        return writer

    # The journal file progress events append to.
    def get_progress_path(self):
        progress_path = self.root / self.cfg.progress_name
        return progress_path

    # Artifact path for one kind and optional item.
    def get_artifact_path(self, kind: str, item: str):
        ext = getattr(self.cfg, kind, self.cfg.default_ext)
        name = f"{kind}{ext}"
        if item:
            name = f"{kind}.{item}{ext}"
        artifact_path = self.artifact_dir / name
        return artifact_path

    # Report whether one artifact already exists.
    def done(self, kind: str, item: str = ""):
        artifact_path = self.get_artifact_path(kind, item)
        exists = artifact_path.is_file()
        return exists

    # Serialize one artifact body to bytes.
    def make_payload(self, obj):
        if isinstance(obj, bytes):
            payload = obj
        elif isinstance(obj, str):
            payload = obj.encode()
        elif isinstance(obj, list):
            text = ""
            for row_item in obj:
                line = json.dumps(row_item, sort_keys=True)
                text += line + "\n"
            payload = text.encode()
        else:
            payload = json.dumps(obj, indent=1, sort_keys=True).encode()
        return payload

    # Store one artifact atomically.
    def write(self, kind: str, obj, item: str = ""):
        artifact_path = self.get_artifact_path(kind, item)
        payload = self.make_payload(obj)
        tmp = artifact_path.with_name(artifact_path.name + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(artifact_path)

    # Append one progress event to the journal.
    def log(self, event: dict):
        line = json.dumps(event, sort_keys=True)
        with self.get_progress_path().open("a") as journal:
            journal.write(line + "\n")

    # Store the run summary beside the artifacts.
    def finish(self, summary_dict: dict):
        text = json.dumps(summary_dict, indent=1, sort_keys=True)
        (self.root / self.cfg.summary_name).write_text(text)
        print(f"summary {self.root / self.cfg.summary_name}")
