from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import audiogen_eval_protocols
from audiogen_eval_protocols import GenArm, LoaderConfig, Protocol, ProtocolLoader


@dataclass(frozen=True)
class MetricWindow:
    seconds: float
    downmix: str
    pad_samples: int
    band_rate: int
    remove_dc: bool
    renorm_gen: bool
    renorm_ref: bool
    peak_dbfs: float
    rms_norm: bool
    rms_dbfs: float


@dataclass(frozen=True)
class FadProbeWindow:
    remove_dc: bool
    pad_samples: int


@dataclass(frozen=True)
class MetricWindowDefaults:
    seconds: float = 10.0
    pad_samples: int = 0
    remove_dc: bool = True
    renorm_gen: bool = True
    renorm_ref: bool = False
    peak_dbfs: float = 0.0
    rms_norm: bool = False
    rms_dbfs: float = -20.0


@dataclass(frozen=True)
class MetricWindowConfig:
    gen_ref: MetricWindowDefaults = MetricWindowDefaults()
    # TangoMusic on MusicCaps levels both sources by RMS
    rms_levelled: MetricWindowDefaults = MetricWindowDefaults(renorm_gen=False, rms_norm=True)
    rms_cell_list: tuple[tuple[str, str], ...] = (("tango-music", "musiccaps"),)
    # native reconstruction keeps the reference level
    recon_native: MetricWindowDefaults = MetricWindowDefaults(renorm_gen=False, rms_norm=False)


@dataclass(frozen=True)
class MetricOptionConfig:
    metric_list: tuple[str, ...] = ("fd_pann", "fad")
    cell_field_list: tuple[str, ...] = ("downmix", "band_rate")


@dataclass(frozen=True)
class EvalRunConfig:
    model: str = ""
    dataset: str = ""
    data_root: Path = Path("")
    out_dir: Path = Path("")
    protocol_dir: str = ""
    evaluation: str = "recon_ref"
    num_clips: int = 0
    batch_size: int = 4
    seed: int = 0
    device: str = "cuda"
    subset: str = ""
    shard: int = 0
    num_shards: int = 1
    metrics: str = ""
    decoder_frames: int = 0
    cfg_scale: float = 0.0
    recon_eps: str = "row"
    recon_level: str = "native"
    base_ckpt: str = ""
    decoder_ckpt: str = ""
    decoder_weights: str = "raw"
    vocoder_ckpt: str = ""
    generation_cache: str = ""
    ref_rows: str = ""
    cache_shard_clips: int = 128
    cache_artifact_dir: str = "artifact"
    cache_dtype: str = "bfloat16"
    gen_dtype: str = ""
    recon_seed_offset: int = 100_000
    seconds: float = 0.0
    arm: GenArm | None = None
    window_dict: dict[str, MetricWindow] | None = None


@dataclass(frozen=True)
class LoadProtocolConfig:
    protocol_file: str = "protocol.yaml"
    family_file: str = "family.yaml"
    metric_file: str = "metrics.yaml"
    eval_file: str = "evaluations.yaml"
    bundled_dir: str = "protocols"


class EvalProtocol:
    def __init__(self, protocol: Protocol):
        self.protocol = protocol
        self.metric_suite = protocol.metric_suite
        self.eval_suite = protocol.eval_suite

    # Load a protocol tree, the package's bundled one by default.
    @classmethod
    def make_from_dir(cls, protocol_dir: str):
        load_cfg = LoadProtocolConfig()
        loader_cfg = LoaderConfig(
            protocol_file=load_cfg.protocol_file,
            family_file=load_cfg.family_file,
            metric_file=load_cfg.metric_file,
            eval_file=load_cfg.eval_file,
        )
        loader = ProtocolLoader(loader_cfg)
        root = Path(protocol_dir)
        if not protocol_dir:
            package_dir = Path(audiogen_eval_protocols.__file__).parent
            root = package_dir / load_cfg.bundled_dir
        protocol = loader.load_protocol(root)
        eval_protocol = cls(protocol)
        return eval_protocol

    # Name the protocol files declare.
    def get_name(self):
        name = self.protocol.name
        return name

    # Protocol cells scored; empty selections score the default.
    def get_cell_list(self, protocol_metric_set: set[str]):
        cell_list = sorted(protocol_metric_set)
        if not cell_list:
            cell_list = [self.eval_suite.default_metric]
        return cell_list

    # The generation arm these metrics score.
    def get_metric_arm(self, model: str, dataset: str, protocol_metric_set: set[str]):
        cell_list = self.get_cell_list(protocol_metric_set)
        cell = self.protocol.get_cell(model, dataset, cell_list[0])
        arm = cell.arm
        return arm

    # Render one boolean as on or off.
    def get_flag_text(self, flag: bool):
        text = "off"
        if flag:
            text = "on"
        return text

    # Report lines for one user-set window.
    def make_window_line_list(self, name: str, window: MetricWindow):
        dc_text = self.get_flag_text(window.remove_dc)
        line_list = [
            f"  {name} window  seconds={window.seconds:g} dc={dc_text} "
            f"pad={window.pad_samples} peak={window.peak_dbfs:g}dBFS"
        ]
        if window.rms_norm:
            line_list.append(f"  {name} rms_norm {window.rms_dbfs:g}dBFS both sources")
        return line_list

    # Lines naming every pinned value this run scores under.
    def make_report_line_list(self, model: str, dataset: str, metric_list,
                              protocol_metric_set: set[str],
                              window_dict: dict[str, MetricWindow]):
        metric_text = ",".join(metric_list) or "(none, default arm)"
        protocol_name = self.get_name()
        line_list = [
            f"protocol {protocol_name}  {model} / {dataset}",
            f"  metrics   {metric_text}",
        ]
        for name in self.get_cell_list(protocol_metric_set):
            cell = self.protocol.get_cell(model, dataset, name)
            arm = cell.arm
            line_list.append(
                f"  {name} arm     steps={arm.steps} cfg={arm.cfg_scale} "
                f"cond=({arm.seconds_start:g}, {arm.seconds_total:g}) "
                f"level_policy={arm.level_policy}"
            )
            window_line_list = self.make_window_line_list(name, window_dict[name])
            line_list.extend(window_line_list)
            if cell.paper_fd != "fd_pann":
                line_list.append(f"  {name} paper_fd {cell.paper_fd}")
        return line_list
