from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

from audiogen_eval_protocols import Cell
import click

from .src.evaluation.evaluation import EvalTaskMaker, EvalPartLoader
from .src.evaluation.registry import (
    EvalProtocol,
    EvalRunConfig,
    MetricOptionConfig,
    MetricWindow,
    MetricWindowConfig,
)
from .src.evaluation.result_writer import EvalResultWriter, EvalWriterConfig
from .src.runtime.dataroot import apply_data_root


class MetricOptionReader:
    def __init__(self, cfg: MetricOptionConfig, default_cfg: MetricWindowConfig):
        self.cfg = cfg
        self.default_cfg = default_cfg

    # Option name for one metric's window field.
    def get_option_name(self, metric: str, field_name: str):
        option_name = f"{metric}_{field_name}"
        return option_name

    # Code default window for this run's lane and cell.
    def get_window_defaults(self, cfg: EvalRunConfig, family: str):
        defaults = self.default_cfg.gen_ref
        if (family, cfg.dataset) in self.default_cfg.rms_cell_list:
            defaults = self.default_cfg.rms_levelled
        if cfg.evaluation == "recon_ref" and cfg.recon_level == "native":
            defaults = self.default_cfg.recon_native
        return defaults

    # One window value: cell, then flag, then code default.
    def get_window_value(self, option_dict: dict, cell: Cell, defaults, metric: str,
                         field_name: str):
        if field_name in self.cfg.cell_field_list:
            value = getattr(cell, field_name)
        else:
            option_name = self.get_option_name(metric, field_name)
            value = option_dict[option_name]
            if value is None:
                value = getattr(defaults, field_name)
        return value

    # Per-metric window: channel and band from cell, rest flags.
    def make_window_dict(self, option_dict: dict, eval_protocol: EvalProtocol,
                         cfg: EvalRunConfig):
        window_dict = {}
        field_tuple = fields(MetricWindow)
        family = eval_protocol.protocol.get_family(cfg.model)
        defaults = self.get_window_defaults(cfg, family)
        for metric_item in self.cfg.metric_list:
            cell = eval_protocol.protocol.get_cell(cfg.model, cfg.dataset, metric_item)
            value_dict = {}
            for field_item in field_tuple:
                value = self.get_window_value(
                    option_dict, cell, defaults, metric_item, field_item.name,
                )
                value_dict[field_item.name] = value
            window_dict[metric_item] = MetricWindow(**value_dict)
        return window_dict

    # Every option that is not a window field.
    def make_run_dict(self, option_dict: dict):
        window_name_set = set()
        field_tuple = fields(MetricWindow)
        for metric_item in self.cfg.metric_list:
            for field_item in field_tuple:
                option_name = self.get_option_name(metric_item, field_item.name)
                window_name_set.add(option_name)
        run_dict = {}
        for key_item, value_item in option_dict.items():
            if key_item not in window_name_set:
                run_dict[key_item] = value_item
        return run_dict


class EvalRunPlanner:
    def __init__(self, eval_protocol: EvalProtocol):
        self.eval_protocol = eval_protocol

    # The evaluation spec, with an optional metric override.
    def get_spec(self, cfg: EvalRunConfig):
        spec = self.eval_protocol.eval_suite.get_eval_spec(cfg.evaluation)
        if cfg.metrics:
            metric_tuple = tuple(cfg.metrics.split(","))
            spec = replace(spec, metric_list=metric_tuple)
        return spec

    # Pin the protocol arm and the default metric's seconds.
    def apply_protocol_config(self, cfg: EvalRunConfig, spec):
        metric_suite = self.eval_protocol.metric_suite
        protocol_metric_set = metric_suite.get_protocol_metric_set(spec.metric_list)
        window = cfg.window_dict[self.eval_protocol.eval_suite.default_metric]
        arm = self.eval_protocol.get_metric_arm(cfg.model, cfg.dataset, protocol_metric_set)
        resolved = replace(cfg, seconds=window.seconds, arm=arm)
        return resolved

    # Print every pinned value this run scores under.
    def log_report(self, cfg: EvalRunConfig, spec):
        metric_suite = self.eval_protocol.metric_suite
        protocol_metric_set = metric_suite.get_protocol_metric_set(spec.metric_list)
        line_list = self.eval_protocol.make_report_line_list(
            cfg.model, cfg.dataset, spec.metric_list, protocol_metric_set, cfg.window_dict)
        for line in line_list:
            print(line)


@click.command()
@click.option("--model", required=True, help="a model some protocol family lists")
@click.option("--dataset", required=True, help="a dataset the protocol pins")
@click.option("--data_root", type=Path, required=True, help="eval data, weights and models")
@click.option("--out_dir", type=Path, required=True, help="artifacts and summary")
@click.option("--protocol_dir", default=EvalRunConfig.protocol_dir,
              help="protocol YAML root; empty = the tree bundled with the package", hidden=True)
@click.option("--evaluation", default=EvalRunConfig.evaluation,
              help="an evaluation the protocol suite names")
@click.option("--num_clips", type=int, default=EvalRunConfig.num_clips,
              help="0 = the whole population")
@click.option("--batch_size", type=int, default=EvalRunConfig.batch_size)
@click.option("--seed", type=int, default=EvalRunConfig.seed, hidden=True)
@click.option("--device", default=EvalRunConfig.device)
@click.option("--subset", default=EvalRunConfig.subset,
              help="override the dataset subset, 'all' = every split", hidden=True)
@click.option("--shard", type=int, default=EvalRunConfig.shard, hidden=True)
@click.option("--num_shards", type=int, default=EvalRunConfig.num_shards,
              help="workers producing wavs, >1 skips metrics", hidden=True)
@click.option("--metrics", default=EvalRunConfig.metrics, help="comma list of metric suite names")
@click.option("--decoder_frames", type=int, default=EvalRunConfig.decoder_frames,
              help="0 = protocol default", hidden=True)
@click.option("--cfg_scale", type=float, default=EvalRunConfig.cfg_scale,
              help="0 = the protocol arm", hidden=True)
@click.option("--recon_eps", type=click.Choice(["fresh", "row"]), default=EvalRunConfig.recon_eps, hidden=True)
@click.option("--recon_level", type=click.Choice(["native", "upstream"]),
              default=EvalRunConfig.recon_level, hidden=True)
@click.option("--base_ckpt", required=True, help="pretrained checkpoint the codec loads")
@click.option("--decoder_ckpt", default=EvalRunConfig.decoder_ckpt)
@click.option("--decoder_weights", type=click.Choice(["ema", "raw"]),
              default=EvalRunConfig.decoder_weights, hidden=True)
@click.option("--vocoder_ckpt", default=EvalRunConfig.vocoder_ckpt)
@click.option("--generation_cache", default=EvalRunConfig.generation_cache)
@click.option("--cache_artifact_dir", default=EvalRunConfig.cache_artifact_dir,
              help="artifact folder name inside --generation_cache")
@click.option("--ref_rows", default=EvalRunConfig.ref_rows, hidden=True)
@click.option("--cache_shard_clips", type=int, default=EvalRunConfig.cache_shard_clips, hidden=True)
@click.option("--cache_dtype", type=click.Choice(["bfloat16", "float16", "float32"]),
              default=EvalRunConfig.cache_dtype, hidden=True)
@click.option("--gen_dtype", type=click.Choice(["", "float16", "bfloat16", "float32"]),
              default=EvalRunConfig.gen_dtype, hidden=True)
@click.option("--fd_pann_seconds", type=float, default=None, hidden=True)
@click.option("--fd_pann_pad_samples", type=int, default=None, hidden=True)
@click.option("--fd_pann_remove_dc", type=bool, default=None, hidden=True)
@click.option("--fd_pann_renorm_gen", type=bool, default=None, hidden=True)
@click.option("--fd_pann_renorm_ref", type=bool, default=None, hidden=True)
@click.option("--fd_pann_peak_dbfs", type=float, default=None, hidden=True)
@click.option("--fd_pann_rms_norm", type=bool, default=None, hidden=True)
@click.option("--fd_pann_rms_dbfs", type=float, default=None, hidden=True)
@click.option("--fad_seconds", type=float, default=None, hidden=True)
@click.option("--fad_pad_samples", type=int, default=None, hidden=True)
@click.option("--fad_remove_dc", type=bool, default=None, hidden=True)
@click.option("--fad_renorm_gen", type=bool, default=None, hidden=True)
@click.option("--fad_renorm_ref", type=bool, default=None, hidden=True)
@click.option("--fad_peak_dbfs", type=float, default=None, hidden=True)
@click.option("--fad_rms_norm", type=bool, default=None, hidden=True)
@click.option("--fad_rms_dbfs", type=float, default=None, hidden=True)
def main(**option_dict):
    window_option_cfg = MetricOptionConfig()
    window_default_cfg = MetricWindowConfig()
    window_reader = MetricOptionReader(window_option_cfg, window_default_cfg)
    run_dict = window_reader.make_run_dict(option_dict)
    cfg = EvalRunConfig(**run_dict)
    eval_protocol = EvalProtocol.make_from_dir(cfg.protocol_dir)
    window_dict = window_reader.make_window_dict(option_dict, eval_protocol, cfg)
    cfg = replace(cfg, window_dict=window_dict)
    planner = EvalRunPlanner(eval_protocol)
    spec = planner.get_spec(cfg)
    apply_data_root(cfg.data_root)
    cfg = planner.apply_protocol_config(cfg, spec)
    planner.log_report(cfg, spec)
    loader = EvalPartLoader(cfg.data_root, eval_protocol)
    maker = EvalTaskMaker(loader)
    task = maker.make_task(cfg, spec, eval_protocol)
    writer = EvalResultWriter.make_from_dir(EvalWriterConfig(), cfg.out_dir)
    summary_dict = task.run_evaluation(writer)
    writer.finish(summary_dict)


if __name__ == "__main__":
    main()
