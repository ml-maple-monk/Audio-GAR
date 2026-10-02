from __future__ import annotations

import io
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import click
import torch
from tqdm import tqdm

from .src.dataset.registry import SpecTrainData, TrainWriterConfig, load_eval_exclusions
from .src.dataset.training_loader import (
    ApplyChannelMix,
    DecoderLatentDataset,
    LatentAugmentDataset,
    ShardAudioSet,
)
from .src.evaluation.registry import FadProbeWindow
from .src.evaluation.result_writer import EvalResultWriter
from .src.models.stable_audio_open.registry import (
    SpecTrainTracking,
    SpecVaeTraining,
    TrainFinetuneConfig,
    get_train_tracking,
    get_vae_training,
)
from .src.models.tango2.registry import (
    RECIPES,
    SpecJointScopes,
    SpecJointWindow,
    SpecMelVaeTraining,
    SpecVocoderTraining,
    TrainJointConfig,
    get_mel_vae_training,
    get_vocoder_training,
)
from .src.models.tokenizer import get_family_ckpt_path, load_tokenizer
from .src.runtime.dataroot import apply_data_root, get_data_root
from .src.train.logging import TrainLogger
from .src.train.loss_function import (
    DiscriminatorLossTerms,
    JointGeneratorLoss,
    LossFactory,
    MelCodecView,
    MelGradTransform,
    VocoderLossSet,
)
from .src.train.train_joint_vocoder import (
    GradNormReader,
    JointOptimizerFactory,
    JointScopeFreezer,
    JointStepFnFactory,
    JointStepFns,
    JointStepRunner,
    JointWeightLoader,
    JointWindowCutter,
    MelDecoderFactory,
    VocoderGeneratorFactory,
    VocoderJointModules,
    VocoderTrainModules,
)
from .src.train.train_vae import (
    CkptBlobLoader,
    FadProbe,
    VaeOptimizerFactory,
    VaeStepRunner,
    VaeTrainModules,
)
from .src.train.vendor.hifigan.models import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)


@dataclass(frozen=True)
class SpecFinetune:
    generator: str
    cache_dirs: tuple[str, ...]
    training: SpecVaeTraining
    tracking: SpecTrainTracking
    probe_window: FadProbeWindow
    channel_augment: bool
    decoder_start: int
    seed: int
    device: str


@dataclass(frozen=True)
class SpecJointVocoder:
    generator: str
    cache_dirs: tuple[str, ...]
    training: SpecMelVaeTraining
    vocoder: SpecVocoderTraining
    tracking: SpecTrainTracking
    exclusions: frozenset
    seed: int
    device: str
    vocoder_loss_weight: float
    vocoder_crop_frames: int
    vocoder_windows_per_clip: int
    decoder_train: str
    vocoder_train: str
    init_from: str
    critics_only: bool
    vocoder_init: str
    init_critics_from: str


@dataclass
class DataSplit:
    train_set: DecoderLatentDataset
    test_set: DecoderLatentDataset


@dataclass
class TrainParts:
    modules: object
    optimizers: object


@dataclass
class FinetuneTools:
    step_runner: VaeStepRunner
    optimizer_factory: VaeOptimizerFactory
    fad_probe: FadProbe


@dataclass
class JointTools:
    step_runner: JointStepRunner
    optimizer_factory: JointOptimizerFactory
    fns: JointStepFns


class CacheItemSeed:
    def __init__(self, seed: int, offset: int):
        self.seed = seed
        self.offset = offset

    # One micro-batch seed, offset like the kernel run.
    def get_item_seed(self, idx: int):
        item_seed = self.seed + self.offset + idx
        return item_seed

    # Seeds for every micro-batch the fit draws.
    def make_item_seeds(self, n_items: int):
        seed_list = []
        for item_idx in range(n_items):
            item_seed = self.get_item_seed(item_idx)
            seed_list.append(item_seed)
        item_seeds = tuple(seed_list)
        return item_seeds


class CacheBuilder:
    def __init__(self, cfg, writer: EvalResultWriter, item_seed: CacheItemSeed):
        self.cfg = cfg
        self.writer = writer
        self.item_seed = item_seed

    # Sorted non-empty cache dirs from the comma list.
    def get_cache_dirs(self, text: str):
        dir_list = []
        for part_item in text.split(","):
            if part_item:
                dir_list.append(part_item)
        dir_list.sort()
        cache_dirs = tuple(dir_list)
        return cache_dirs

    # Cache dirs as paths for the loader calls.
    def get_cache_paths(self, cache_dirs: tuple):
        path_list = []
        for dir_item in cache_dirs:
            dir_path = Path(dir_item)
            path_list.append(dir_path)
        return path_list

    # Cache folder names, logged with the params.
    def get_cache_slots(self, cache_dirs: tuple):
        slot_list = []
        for dir_item in cache_dirs:
            dir_path = Path(dir_item)
            slot_list.append(dir_path.name)
        return slot_list

    # Header per cache dir; caches keep their own clip length.
    def load_cache_headers(self, cache_dirs: tuple):
        header_dict = {}
        for dir_item in cache_dirs:
            dir_path = Path(dir_item)
            header = DecoderLatentDataset.load_cache_header(dir_path)
            header_dict[str(dir_path)] = header
        return header_dict

    # Debug folder for the mlflow db and probe files.
    def open_debug_dir(self):
        debug_dir = self.writer.root / TrainWriterConfig.debug_dir
        debug_dir.mkdir(parents=True, exist_ok=True)
        return debug_dir


class CacheTrainer:
    # Micro-batches in index order for every fit step.
    def make_fit_loader(self):
        fit = self.spec.training.fit
        runtime = self.spec.training.runtime
        workers = runtime.data_workers
        stop_idx = fit.steps * fit.accumulation
        item_range = range(stop_idx)
        window_set = torch.utils.data.Subset(self.clip_set, item_range)
        prefetch = None
        if workers:
            prefetch = max(1, runtime.data_prefetch // workers)
        loader = torch.utils.data.DataLoader(
            window_set,
            batch_size=None,
            # __getitem__ owns the randomness; index order must stay
            shuffle=False,
            num_workers=workers,
            prefetch_factor=prefetch,
            pin_memory=self.spec.device != "cpu",
            persistent_workers=workers > 0,
        )
        return loader

    # Save one bundle as the step-named checkpoint artifact.
    def write_ckpt(self, kind: str, bundle: dict, step: int):
        buffer = io.BytesIO()
        torch.save(bundle, buffer)
        payload = buffer.getvalue()
        self.writer.write(kind, payload, item=f"step{step + 1:07d}")


class TrainDecoderFinetune(CacheTrainer):
    def __init__(
        self,
        spec: SpecFinetune,
        codec,
        data_split: DataSplit,
        parts: TrainParts,
        tools: FinetuneTools,
        logger: TrainLogger,
        writer: EvalResultWriter,
        debug_dir: Path,
    ):
        self.spec = spec
        self.codec = codec
        self.clip_set = data_split.train_set
        self.test_set = data_split.test_set
        self.modules = parts.modules
        self.optimizers = parts.optimizers
        self.tools = tools
        self.logger = logger
        self.writer = writer
        self.debug_dir = debug_dir
        self.pending_list = []

    # Updates one lane has taken through idx; offset aware.
    def get_lane_step(self, idx: int, lane: str):
        decoder_start = self.spec.decoder_start
        if lane == "decoder":
            lane_step = idx // 2 + 1 - (decoder_start + 1) // 2
        elif idx < decoder_start:
            lane_step = idx + 1
        else:
            lane_step = decoder_start + (idx + 1) // 2 - decoder_start // 2
        return lane_step

    # Journal and mlflow rows once a checkpoint covers them.
    def write_pending_rows(self):
        fit = self.spec.training.fit
        for row_item in self.pending_list:
            self.writer.log(row_item)
            lane_step = self.get_lane_step(row_item["index"], row_item["lane"])
            if lane_step % fit.logging_every == 0 or row_item["index"] + 1 == fit.steps:
                self.logger.log_step(row_item)
        self.pending_list = []

    # Buffer one step row until the next checkpoint.
    def log_metrics(self, row: dict):
        self.pending_list.append(row)

    # Held-out tables plus FAD probe, then a tracking snapshot.
    def run_eval(self, step: int, elapsed: float):
        spec = self.spec
        tracking = spec.tracking
        tables = self.tools.step_runner.run_holdout(
            self.codec, self.test_set, tracking,
            tracking.eval_batch_clips, spec.training.fit.crop_frames, spec.device,
        )
        fad_table = self.tools.fad_probe.run(
            self.modules, self.codec, self.test_set, tracking.metrics, tracking.fad_pairs,
            self.debug_dir,
        )
        tables["fad"] = fad_table
        self.logger.log_eval(step, elapsed, tables)
        snapshot = self.logger.make_snapshot()
        self.writer.write("train_tracking", snapshot, item=f"step{step + 1:07d}")

    # Critic leads until decoder_start, then the lanes alternate.
    def train_step(self, group: tuple, step: int):
        spec = self.spec
        # discriminator and decoder never step on the same batch
        if step < spec.decoder_start or step % 2:
            result = self.tools.step_runner.train_discriminator(self.optimizers, group)
        else:
            result = self.tools.step_runner.train_decoder(self.optimizers, group)
        # both lanes share one clock, so warmup counts global steps
        self.optimizers.decoder_schedule.step()
        self.optimizers.discriminator_schedule.step()
        return result

    # Step, checkpoint, flush rows, evaluate; then write the summary.
    def run(self):
        started_at = time.monotonic()
        spec = self.spec
        fit = spec.training.fit
        loader = self.make_fit_loader()
        last = {}
        group_list = []
        step = 0
        with tqdm(total=fit.steps, unit="step") as bar:
            for batch_data in loader:
                batch_device = DecoderLatentDataset.load_batch_device(
                    batch_data, self.codec, spec.device
                )
                group_list.append(batch_device)
                if len(group_list) < fit.accumulation:
                    continue
                group = tuple(group_list)
                result = self.train_step(group, step)
                group_list = []
                elapsed = time.monotonic() - started_at
                row = {**result, "index": step, "elapsed_seconds": elapsed}
                self.log_metrics(row)
                total = round(row["total"], 4)
                bar.set_postfix(total=total)
                bar.update()
                last = row
                if (step + 1) % fit.checkpoint_every == 0 or step + 1 == fit.steps:
                    bundle = self.tools.optimizer_factory.make_ckpt_bundle(
                        self.modules, self.optimizers, step,
                    )
                    self.write_ckpt("decoder_ckpt", bundle, step)
                    self.write_pending_rows()
                if (step + 1) % fit.eval_every == 0 or step + 1 == fit.steps:
                    eval_elapsed = time.monotonic() - started_at
                    self.run_eval(step, eval_elapsed)
                step += 1
        self.logger.finish()
        snapshot = self.logger.make_snapshot()
        self.writer.write("train_tracking", snapshot, item="final")
        last_index = last.get("index", -1)
        level_list = list(fit.levels)
        summary = {
            "generator": spec.generator,
            "steps": last_index + 1,
            "target_steps": fit.steps,
            "clips": len(self.clip_set.rows),
            "levels": level_list,
            "final_total": last.get("total"),
            "decoder_start": spec.decoder_start,
        }
        self.writer.finish(summary)


class TrainFinetuneBuilder(CacheBuilder):
    # Fit, optimizer and loader controls for one level.
    def make_training_spec(self, generator: str):
        cfg = self.cfg
        training = get_vae_training(generator)
        level = DecoderLatentDataset.make_level_item(cfg.level)
        # finetuning is single level; the tuple keeps the config shape
        fit = replace(
            training.fit,
            levels=(level,),
            steps=cfg.max_train_steps,
            batch_clips=cfg.per_device_train_batch_size,
            accumulation=cfg.gradient_accumulation_steps,
            crop_frames=cfg.crop_frames,
            checkpoint_every=cfg.checkpointing_steps,
            logging_every=cfg.logging_steps,
            eval_every=cfg.eval_steps,
        )
        optim = replace(
            training.optim,
            decoder_lr=cfg.learning_rate,
            discriminator_lr=cfg.discriminator_learning_rate,
            total_steps=cfg.max_train_steps,
        )
        runtime = replace(
            training.runtime,
            data_workers=cfg.data_workers,
            data_prefetch=cfg.data_prefetch,
        )
        training = replace(training, fit=fit, optim=optim, runtime=runtime)
        return training

    # Resolved finetune spec bound to the cache bundle.
    def make_spec(self, cache_dirs: tuple, header: dict):
        cfg = self.cfg
        generator = header["generator"]
        training = self.make_training_spec(generator)
        tracking = get_train_tracking()
        probe_window = FadProbeWindow(
            remove_dc=cfg.probe_remove_dc, pad_samples=cfg.probe_pad_samples)
        spec = SpecFinetune(
            generator=generator,
            cache_dirs=cache_dirs,
            training=training,
            tracking=tracking,
            probe_window=probe_window,
            channel_augment=cfg.channel_augment,
            decoder_start=cfg.decoder_start,
            seed=cfg.seed,
            device=cfg.device,
        )
        return spec

    # Pinned pretrained codec on the run device.
    def open_codec(self, spec: SpecFinetune):
        ckpt_path = get_family_ckpt_path(spec.generator)
        codec = load_tokenizer(
            spec.generator, ckpt_path=ckpt_path, device=spec.device, load_pretrained=True,
        )
        return codec

    # Index cached clips at the level, split held-out clips.
    def open_clip_sets(self, spec: SpecFinetune, codec, header: dict):
        fit = spec.training.fit
        runtime = spec.training.runtime
        cache_paths = self.get_cache_paths(spec.cache_dirs)
        rows = DecoderLatentDataset.collect_latents(
            cache_paths, fit.levels[0], min_frames=fit.crop_frames,
        )
        train_rows, test_rows = DecoderLatentDataset.split_train_eval(spec.tracking, rows)
        header_dict = self.load_cache_headers(spec.cache_dirs)
        data_root = get_data_root()
        audio = ShardAudioSet(data_root)
        item_seeds = self.item_seed.make_item_seeds(fit.steps * fit.accumulation)
        data_spec = SpecTrainData(
            batch_clips=fit.batch_clips,
            accumulation=fit.accumulation,
            crop_frames=fit.crop_frames,
            sample_size=fit.sample_size,
            data_workers=runtime.data_workers,
            data_prefetch=runtime.data_prefetch,
        )
        if spec.channel_augment:
            channel_mix = ApplyChannelMix()
            train_set = LatentAugmentDataset(
                audio, header, train_rows, codec.audio_channels,
                spec=data_spec, item_seeds=item_seeds, headers=header_dict,
                transforms=(channel_mix,),
            )
        else:
            train_set = DecoderLatentDataset(
                audio, header, train_rows, codec.audio_channels,
                spec=data_spec, item_seeds=item_seeds, headers=header_dict,
            )
        test_set = DecoderLatentDataset(
            audio, header, test_rows, codec.audio_channels,
            spec=data_spec, headers=header_dict,
        )
        data_split = DataSplit(train_set=train_set, test_set=test_set)
        return data_split

    # Compiled decoder, seeded critic, optimizers over restored weights.
    def make_components(self, spec: SpecFinetune, codec):
        vae_model = codec.get_vae_module()
        vae_model.requires_grad_(False)
        vae_model.eval()
        decoder_model = vae_model.decoder
        decoder_model.requires_grad_(True)
        decoder_model.train()
        # compiled state dicts carry an _orig_mod prefix
        decoder_model = torch.compile(decoder_model)
        init_seed = self.item_seed.get_item_seed(spec.training.fit.init_seed_index)
        # the released discriminator weights never shipped, so seed one
        loss_factory = LossFactory(spec.training.loss)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            discriminator_model = loss_factory.make_discriminator_model(codec.audio_channels)
        discriminator_model = discriminator_model.to(spec.device)
        stft_model = loss_factory.make_stft_reconstruction(codec.sample_rate)
        stft_model.sum_difference.to(spec.device)
        stft_model.channel.to(spec.device)
        modules = VaeTrainModules(decoder_model, discriminator_model, stft_model)
        optimizer_factory = VaeOptimizerFactory(spec.training.optim)
        optimizers = optimizer_factory.make_optimizers(modules, spec.decoder_start)
        parts = TrainParts(modules=modules, optimizers=optimizers)
        return parts

    # Step runner, optimizer factory and FAD probe for the lane.
    def make_tools(self, spec: SpecFinetune, parts: TrainParts):
        loss_terms = DiscriminatorLossTerms()
        step_runner = VaeStepRunner(parts.modules, spec.training.loss, loss_terms)
        optimizer_factory = VaeOptimizerFactory(spec.training.optim)
        fad_probe = FadProbe(spec.probe_window, spec.device)
        tools = FinetuneTools(
            step_runner=step_runner, optimizer_factory=optimizer_factory, fad_probe=fad_probe,
        )
        return tools

    # Minimal mlflow params: model, data, fit and optimizer.
    def make_param_dict(self, spec: SpecFinetune):
        cache_slots = self.get_cache_slots(spec.cache_dirs)
        param_dict = {
            "generator": spec.generator,
            "cache_slots": cache_slots,
            "fit": asdict(spec.training.fit),
            "optimizer": asdict(spec.training.optim),
        }
        return param_dict

    # mlflow logger under debug, named after the output dir.
    def open_tracking(self, spec: SpecFinetune, debug_dir: Path):
        db_path = debug_dir / spec.tracking.db_name
        logger = TrainLogger(db_path, spec.tracking, self.writer.root.name)
        param_dict = self.make_param_dict(spec)
        logger.log_params(param_dict)
        return logger

    # Build spec, codec, data and nets, then the trainer.
    def make_trainer(self):
        torch.backends.cudnn.benchmark = True
        cache_dirs = self.get_cache_dirs(self.cfg.cache_dirs)
        first_path = Path(cache_dirs[0])
        header = DecoderLatentDataset.load_cache_header(first_path)
        spec = self.make_spec(cache_dirs, header)
        debug_dir = self.open_debug_dir()
        codec = self.open_codec(spec)
        data_split = self.open_clip_sets(spec, codec, header)
        parts = self.make_components(spec, codec)
        tools = self.make_tools(spec, parts)
        logger = self.open_tracking(spec, debug_dir)
        trainer = TrainDecoderFinetune(
            spec, codec, data_split, parts, tools, logger, self.writer, debug_dir,
        )
        return trainer


class TrainJointVocoder(CacheTrainer):
    def __init__(
        self,
        spec: SpecJointVocoder,
        codec,
        clip_set: DecoderLatentDataset,
        parts: TrainParts,
        tools: JointTools,
        logger: TrainLogger,
        writer: EvalResultWriter,
        item_seed: CacheItemSeed,
    ):
        self.spec = spec
        self.codec = codec
        self.clip_set = clip_set
        self.modules = parts.modules
        self.optimizers = parts.optimizers
        self.tools = tools
        self.logger = logger
        self.writer = writer
        self.item_seed = item_seed
        self.pending_list = []

    # Progress-bar fields: loss, critic, and the grad monitors.
    def make_live_row(self, row: dict):
        total = round(row["total"], 4)
        lr_text = f"{row['lr']:.3g}"
        live = {"total": total, "lr": lr_text}
        monitor_pairs = (
            ("vocoder_discriminator", "critic"),
            ("grad_norm_decoder", "gdec"),
            ("grad_norm_vocoder", "gvoc"),
            ("grad_norm_mpd", "gmpd"),
        )
        for key_item, short_item in monitor_pairs:
            if key_item in row:
                live[short_item] = f"{row[key_item]:.3g}"
        return live

    # Journal and mlflow rows once a checkpoint covers them.
    def write_pending_rows(self):
        fit = self.spec.training.fit
        for row_item in self.pending_list:
            self.writer.log(row_item)
            step_count = row_item["index"] + 1
            if step_count % fit.logging_every == 0 or step_count == fit.steps:
                self.logger.log_step(row_item)
        self.pending_list = []

    # Buffer one step row until the next checkpoint.
    def log_metrics(self, row: dict):
        self.pending_list.append(row)

    # One joint update on a seeded window draw.
    def train_step(self, group: tuple, step: int):
        spec = self.spec
        fit = spec.training.fit
        window_idx = fit.steps * fit.accumulation + step
        window_seed = self.item_seed.get_item_seed(window_idx)
        result = self.tools.step_runner.run_step(
            self.optimizers, group, step, spec.vocoder_crop_frames,
            spec.vocoder_windows_per_clip, window_seed,
        )
        return result

    # Step, guard, checkpoint, flush rows; then write the summary.
    def run(self):
        started_at = time.monotonic()
        fit = self.spec.training.fit
        # recompiles are forbidden once step three is reached
        fence = 3
        loader = self.make_fit_loader()
        last = {}
        group_list = []
        step = 0
        with tqdm(total=fit.steps, unit="step") as bar:
            for batch_data in loader:
                batch_device = DecoderLatentDataset.load_batch_device(
                    batch_data, self.codec, self.spec.device
                )
                group_list.append(batch_device)
                if len(group_list) < fit.accumulation:
                    continue
                group = tuple(group_list)
                result = self.train_step(group, step)
                group_list = []
                elapsed = time.monotonic() - started_at
                row = {**result, "index": step, "elapsed_seconds": elapsed}
                self.log_metrics(row)
                live = self.make_live_row(row)
                bar.set_postfix(live)
                bar.update()
                last = row
                if (step + 1) % fit.checkpoint_every == 0 or step + 1 == fit.steps:
                    bundle = self.tools.optimizer_factory.make_ckpt_bundle(
                        self.modules, self.optimizers, step,
                    )
                    self.write_ckpt("vae_ckpt", bundle, step)
                    self.write_pending_rows()
                    snapshot = self.logger.make_snapshot()
                    self.writer.write("train_tracking", snapshot, item=f"step{step + 1:07d}")
                self.tools.fns.apply_compile_fence(step, fence)
                step += 1
        self.logger.finish()
        last_index = last.get("index", -1)
        summary = {"steps": last_index + 1, "final_total": last.get("total")}
        self.writer.finish(summary)


class TrainJointBuilder(CacheBuilder):
    # Recipe value unless the flag overrides it.
    def get_option_value(self, value, fallback):
        option_value = value
        if value is None:
            option_value = fallback
        return option_value

    # Blob labels for one level or a comma list.
    def make_level_items(self, text: str):
        level_list = []
        for token_item in text.split(","):
            if token_item.strip():
                level_item = DecoderLatentDataset.make_level_item(token_item)
                level_list.append(level_item)
        levels = tuple(level_list)
        return levels

    # Recipe controls with flag overrides; peak norm always on.
    def make_training_spec(self, generator: str):
        cfg = self.cfg
        training = get_mel_vae_training(generator, cfg.recipe)
        fit = training.fit
        level_text = self.get_option_value(cfg.level, fit.levels[0])
        level_str = str(level_text)
        levels = self.make_level_items(level_str)
        steps = self.get_option_value(cfg.max_train_steps, fit.steps)
        batch_clips = self.get_option_value(cfg.per_device_train_batch_size, fit.batch_clips)
        accumulation = self.get_option_value(cfg.gradient_accumulation_steps, fit.accumulation)
        crop_frames = self.get_option_value(cfg.crop_frames, fit.crop_frames)
        checkpoint_every = self.get_option_value(cfg.checkpointing_steps, fit.checkpoint_every)
        logging_every = self.get_option_value(cfg.logging_steps, fit.logging_every)
        fit = replace(
            fit,
            levels=levels,
            steps=steps,
            batch_clips=batch_clips,
            accumulation=accumulation,
            crop_frames=crop_frames,
            checkpoint_every=checkpoint_every,
            logging_every=logging_every,
            peak_norm=True,
        )
        data_workers = self.get_option_value(cfg.data_workers, training.runtime.data_workers)
        data_prefetch = self.get_option_value(cfg.data_prefetch, training.runtime.data_prefetch)
        runtime = replace(
            training.runtime,
            data_workers=data_workers,
            data_prefetch=data_prefetch,
        )
        if cfg.precision == "fp32":
            runtime = replace(runtime, autocast_dtype="float32", allow_tf32=False)
        if cfg.precision == "fp16":
            runtime = replace(runtime, autocast_dtype="float16")
        lr = self.get_option_value(cfg.learning_rate, training.optim.lr)
        optim = replace(training.optim, lr=lr)
        training = replace(training, fit=fit, optim=optim, runtime=runtime)
        return training

    # Critic recipe with this run's lr and schedule.
    def make_vocoder_spec(self, generator: str):
        cfg = self.cfg
        # the vocoder recipe never followed --recipe; kept as shipped
        vocoder = get_vocoder_training(generator)
        lr = self.get_option_value(cfg.vocoder_learning_rate, vocoder.optim.lr)
        optim = replace(vocoder.optim, lr=lr, lr_schedule=cfg.vocoder_lr_schedule)
        vocoder = replace(vocoder, optim=optim)
        return vocoder

    # Eval exclusions, empty off wavcaps caches.
    def load_exclusions(self, cache_dirs: tuple):
        cache_paths = self.get_cache_paths(cache_dirs)
        exclusions = frozenset()
        if DecoderLatentDataset.check_wavcaps_cache(cache_paths):
            data_root = get_data_root()
            exclusions = load_eval_exclusions(data_root)
        return exclusions

    # Resolved joint spec bound to the cache bundle.
    def make_spec(self, cache_dirs: tuple, header: dict):
        cfg = self.cfg
        generator = header["generator"]
        training = self.make_training_spec(generator)
        exclusions = self.load_exclusions(cache_dirs)
        vocoder = self.make_vocoder_spec(generator)
        tracking = get_train_tracking()
        spec = SpecJointVocoder(
            generator=generator,
            cache_dirs=cache_dirs,
            training=training,
            vocoder=vocoder,
            tracking=tracking,
            exclusions=exclusions,
            seed=cfg.seed,
            device=cfg.device,
            vocoder_loss_weight=cfg.vocoder_loss_weight,
            vocoder_crop_frames=cfg.vocoder_crop_frames,
            vocoder_windows_per_clip=cfg.vocoder_windows_per_clip,
            decoder_train=cfg.decoder_train,
            vocoder_train=cfg.vocoder_train,
            init_from=cfg.init_from,
            critics_only=cfg.critics_only,
            vocoder_init=cfg.vocoder_init,
            init_critics_from=cfg.init_critics_from,
        )
        return spec

    # Pinned pretrained mel codec on the run device.
    def open_codec(self, spec: SpecJointVocoder):
        ckpt_path = get_family_ckpt_path(spec.generator) or ""
        tokenizer = load_tokenizer(
            spec.generator, ckpt_path=ckpt_path, device=spec.device, load_pretrained=True,
        )
        codec = tokenizer.model
        return codec

    # Every clip once per level; batches draw uniformly over rows.
    def collect_level_rows(self, spec: SpecJointVocoder):
        fit = spec.training.fit
        cache_paths = self.get_cache_paths(spec.cache_dirs)
        row_list = []
        for level_item in fit.levels:
            level_rows = DecoderLatentDataset.collect_latents(
                cache_paths, level_item, min_frames=fit.crop_frames, exclusions=spec.exclusions,
            )
            row_list.extend(level_rows)
        return row_list

    # Training clip set over every requested level.
    def open_clip_set(self, spec: SpecJointVocoder, codec, header: dict):
        fit = spec.training.fit
        runtime = spec.training.runtime
        rows = self.collect_level_rows(spec)
        header_dict = self.load_cache_headers(spec.cache_dirs)
        item_seeds = self.item_seed.make_item_seeds(fit.steps * fit.accumulation)
        sample_size = fit.crop_frames * codec.spec.downsampling_ratio
        data_spec = SpecTrainData(
            batch_clips=fit.batch_clips,
            accumulation=fit.accumulation,
            crop_frames=fit.crop_frames,
            sample_size=sample_size,
            data_workers=runtime.data_workers,
            data_prefetch=runtime.data_prefetch,
            peak_norm=fit.peak_norm,
        )
        data_root = get_data_root()
        audio = ShardAudioSet(data_root)
        clip_set = DecoderLatentDataset(
            audio, header, rows, codec.audio_channels,
            spec=data_spec, item_seeds=item_seeds, headers=header_dict,
        )
        return clip_set

    # Weight-normed generator with freshly seeded waveform critics.
    def open_vocoder_modules(self, spec: SpecJointVocoder, codec, decoder_model):
        init_seed = self.item_seed.get_item_seed(spec.vocoder.fit.init_seed_index)
        generator_factory = VocoderGeneratorFactory()
        generator = None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            mpd_model = MultiPeriodDiscriminator()
            msd_model = MultiScaleDiscriminator()
            # a scratch generator draws from the same seeded stream
            if spec.vocoder_init == "scratch":
                generator = generator_factory.make_scratch_generator()
        mpd_model = mpd_model.to(spec.device)
        msd_model = msd_model.to(spec.device)
        if generator is None:
            generator = generator_factory.make_release_generator(codec)
        generator = generator.to(spec.device)
        vocoder_modules = VocoderTrainModules(generator, mpd_model, msd_model, decoder_model)
        return vocoder_modules

    # Bundles from a compiled net carry the wrapper name.
    def make_unprefixed_state(self, state: dict):
        clean_dict = {}
        for key_item, value in state.items():
            clean_key = key_item.replace("_orig_mod.", "")
            clean_dict[clean_key] = value
        return clean_dict

    # Keep weights from a mel-lane bundle, drop its optimizers.
    def apply_init_state(self, state: dict):
        saved = state.get("vocoder") or {}
        vocoder_dict = {}
        for key_item in ("generator", "mpd", "msd"):
            vocoder_dict[key_item] = saved[key_item]
        kept = {"decoder": state["decoder"], "step": -1, "vocoder": vocoder_dict}
        print("init_from: weights loaded, optimizer state restarts", flush=True)
        return kept

    # Seed MPD and MSD from another bundle; nothing else moves.
    def apply_critic_weights(self, modules: VocoderJointModules, state: dict):
        saved = state.get("vocoder") or {}
        vocoder_modules = modules.vocoder_modules
        mpd_state = self.make_unprefixed_state(saved["mpd"])
        msd_state = self.make_unprefixed_state(saved["msd"])
        vocoder_modules.mpd_model.load_state_dict(mpd_state, strict=True)
        vocoder_modules.msd_model.load_state_dict(msd_state, strict=True)
        bundle_step = state.get("step", "?")
        print(f"critics seeded from step {bundle_step} bundle", flush=True)

    # Build nets, restore or seed them, release scopes, then guard.
    def make_components(self, spec: SpecJointVocoder, codec):
        codec.vae.requires_grad_(False)
        codec.vae.eval()
        decoder_factory = MelDecoderFactory()
        decoder_model = decoder_factory.make_decoder(codec)
        vocoder_modules = self.open_vocoder_modules(spec, codec, decoder_model)
        joint_modules = VocoderJointModules(
            decoder_model, vocoder_modules, spec.vocoder_loss_weight
        )
        modules = joint_modules.to(spec.device)
        blob_loader = CkptBlobLoader()
        if spec.init_from:
            init_path = Path(spec.init_from)
            init_blob = blob_loader.load_state(init_path, spec.device)
            init_state = self.apply_init_state(init_blob)
            weight_loader = JointWeightLoader()
            weight_loader.apply_weights(modules, init_state)
        if spec.init_critics_from:
            critic_path = Path(spec.init_critics_from)
            critic_blob = blob_loader.load_state(critic_path, spec.device)
            self.apply_critic_weights(modules, critic_blob)
        scope_freezer = JointScopeFreezer(SpecJointScopes())
        trainable = scope_freezer.apply_scopes(modules, spec.decoder_train, spec.vocoder_train)
        optimizer_factory = JointOptimizerFactory(
            spec.training.optim, spec.vocoder.optim, spec.training.fit.steps,
        )
        optimizers = optimizer_factory.make_optimizers(modules)
        count_dict = {}
        for name_item, key_list in trainable.items():
            count_dict[name_item] = len(key_list)
        self.writer.log({"trainable": count_dict})
        parts = TrainParts(modules=modules, optimizers=optimizers)
        return parts

    # Compiled decode, hifigan losses and the step runner.
    def make_tools(self, spec: SpecJointVocoder, codec, parts: TrainParts):
        modules = parts.modules
        mel_view = MelCodecView(codec)
        mel_grad = MelGradTransform(codec)
        loss_set = VocoderLossSet(modules.vocoder_modules, mel_view, mel_grad, spec.vocoder.loss)
        dtype = spec.training.runtime.autocast_dtype
        autocast_dtype = torch.bfloat16
        if dtype == "float16":
            autocast_dtype = torch.float16
        fn_factory = JointStepFnFactory()
        fns = fn_factory.make_step_fns(
            modules, spec.device, loss_set, dtype != "float32", autocast_dtype,
        )
        window_spec = SpecJointWindow()
        window_cutter = JointWindowCutter(window_spec)
        grad_reader = GradNormReader()
        generator_loss = JointGeneratorLoss(fns.loss_set, fns.autocast, fns.autocast_dtype)
        step_runner = JointStepRunner(
            modules, fns, window_cutter, grad_reader, window_spec, generator_loss,
        )
        optimizer_factory = JointOptimizerFactory(
            spec.training.optim, spec.vocoder.optim, spec.training.fit.steps,
        )
        tools = JointTools(step_runner=step_runner, optimizer_factory=optimizer_factory, fns=fns)
        return tools

    # Minimal mlflow params: model, data, fit and optimizers.
    def make_param_dict(self, spec: SpecJointVocoder):
        cache_slots = self.get_cache_slots(spec.cache_dirs)
        param_dict = {
            "generator": spec.generator,
            "cache_slots": cache_slots,
            "fit": asdict(spec.training.fit),
            "optimizer": asdict(spec.training.optim),
            "vocoder_optimizer": asdict(spec.vocoder.optim),
        }
        return param_dict

    # mlflow logger under debug, named after the output dir.
    def open_tracking(self, spec: SpecJointVocoder, debug_dir: Path):
        db_path = debug_dir / spec.tracking.db_name
        logger = TrainLogger(db_path, spec.tracking, self.writer.root.name)
        param_dict = self.make_param_dict(spec)
        logger.log_params(param_dict)
        return logger

    # Build spec, codec, data and nets, then the trainer.
    def make_trainer(self):
        cache_dirs = self.get_cache_dirs(self.cfg.cache_dirs)
        first_path = Path(cache_dirs[0])
        header = DecoderLatentDataset.load_cache_header(first_path)
        spec = self.make_spec(cache_dirs, header)
        debug_dir = self.open_debug_dir()
        if not spec.training.runtime.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        codec = self.open_codec(spec)
        clip_set = self.open_clip_set(spec, codec, header)
        parts = self.make_components(spec, codec)
        tools = self.make_tools(spec, codec, parts)
        logger = self.open_tracking(spec, debug_dir)
        trainer = TrainJointVocoder(
            spec, codec, clip_set, parts, tools, logger, self.writer, self.item_seed,
        )
        return trainer


@click.group()
def main():
    pass


@main.command("finetune", help="AudioX-MAF decoder fine-tuning on cached latents.")
@click.option("--cache_dirs", required=True, help="comma list of latent_cache run dirs")
@click.option("--data_root", type=Path, required=True, help="training audio, weights and exclusions")
@click.option("--out_dir", type=Path, required=True, help="checkpoints, tracking and summary; use a fresh dir per run")
@click.option("--probe_remove_dc", type=bool, default=TrainFinetuneConfig.probe_remove_dc, hidden=True)
@click.option("--probe_pad_samples", type=int, default=TrainFinetuneConfig.probe_pad_samples, hidden=True)
@click.option("--level", default=TrainFinetuneConfig.level,
              help="one isolated cache condition: ref or a numeric noise level")
@click.option("--max_train_steps", type=int, default=TrainFinetuneConfig.max_train_steps)
@click.option("--per_device_train_batch_size", type=int,
              default=TrainFinetuneConfig.per_device_train_batch_size)
@click.option("--gradient_accumulation_steps", type=int,
              default=TrainFinetuneConfig.gradient_accumulation_steps, hidden=True)
@click.option("--learning_rate", type=float, default=TrainFinetuneConfig.learning_rate)
@click.option("--discriminator_learning_rate", type=float,
              default=TrainFinetuneConfig.discriminator_learning_rate, hidden=True)
@click.option("--crop_frames", type=int, default=TrainFinetuneConfig.crop_frames, hidden=True)
@click.option("--checkpointing_steps", type=int, default=TrainFinetuneConfig.checkpointing_steps,
              help="save a checkpoint every this many steps")
@click.option("--logging_steps", type=int, default=TrainFinetuneConfig.logging_steps,
              help="send training metrics to mlflow every this many steps", hidden=True)
@click.option("--eval_steps", type=int, default=TrainFinetuneConfig.eval_steps,
              help="score the held-out split every this many steps", hidden=True)
@click.option("--data_workers", type=int, default=TrainFinetuneConfig.data_workers,
              help="loader processes decoding audio for the fit batches", hidden=True)
@click.option("--data_prefetch", type=int, default=TrainFinetuneConfig.data_prefetch,
              help="batches held ready across all loader processes", hidden=True)
@click.option("--channel_augment", is_flag=True, default=TrainFinetuneConfig.channel_augment,
              help="random channel views and short-clip loops in training", hidden=True)
@click.option("--decoder_start", type=int, default=TrainFinetuneConfig.decoder_start,
              help="critic-only steps before the decoder starts; odd values round up", hidden=True)
@click.option("--seed", type=int, default=TrainFinetuneConfig.seed)
@click.option("--device", default=TrainFinetuneConfig.device)
def train_finetune(**option_dict):
    cfg = TrainFinetuneConfig(**option_dict)
    apply_data_root(cfg.data_root)
    writer_cfg = TrainWriterConfig()
    writer = EvalResultWriter.make_from_dir(writer_cfg, cfg.out_dir)
    item_seed = CacheItemSeed(cfg.seed, writer_cfg.item_seed_offset)
    builder = TrainFinetuneBuilder(cfg, writer, item_seed)
    trainer = builder.make_trainer()
    trainer.run()


@main.command("joint", help="TangoMusic decoder and HiFi-GAN vocoder, trained jointly.")
@click.option("--cache_dirs", required=True, help="comma list of latent_cache run dirs")
@click.option("--data_root", type=Path, required=True, help="training audio, weights and exclusions")
@click.option("--out_dir", type=Path, required=True, help="checkpoints, tracking and summary; use a fresh dir per run")
@click.option("--recipe", type=click.Choice(RECIPES), default=TrainJointConfig.recipe, hidden=True)
@click.option("--level", default=TrainJointConfig.level,
              help="ref or a noise level; a comma list mixes levels over rows")
@click.option("--max_train_steps", type=int, default=TrainJointConfig.max_train_steps)
@click.option("--per_device_train_batch_size", type=int,
              default=TrainJointConfig.per_device_train_batch_size)
@click.option("--gradient_accumulation_steps", type=int,
              default=TrainJointConfig.gradient_accumulation_steps, hidden=True)
@click.option("--learning_rate", type=float, default=TrainJointConfig.learning_rate,
              help="decoder and vocoder generator lr; they share one optimizer")
@click.option("--vocoder_learning_rate", type=float, default=TrainJointConfig.vocoder_learning_rate,
              help="MPD and MSD lr; unset follows the vocoder recipe", hidden=True)
@click.option("--vocoder_loss_weight", type=float, default=TrainJointConfig.vocoder_loss_weight,
              help="scale on the summed hifigan generator losses", hidden=True)
@click.option("--vocoder_crop_frames", type=int, default=TrainJointConfig.vocoder_crop_frames,
              help="latent frames per waveform window; 0 = whole crop", hidden=True)
@click.option("--vocoder_windows_per_clip", type=int,
              default=TrainJointConfig.vocoder_windows_per_clip,
              help="random windows drawn per decoded clip", hidden=True)
@click.option("--vocoder_lr_schedule", type=click.Choice(["exponential", "linear"]),
              default=TrainJointConfig.vocoder_lr_schedule, hidden=True)
@click.option("--decoder_train", type=click.Choice(["all", "first", "none"]),
              default=TrainJointConfig.decoder_train,
              help="decoder weights released: all, first (post_quant_conv and conv_in), none", hidden=True)
@click.option("--vocoder_train", type=click.Choice(["all", "last", "none"]),
              default=TrainJointConfig.vocoder_train,
              help="vocoder weights released: all, last (conv_post), none", hidden=True)
@click.option("--vocoder_init", type=click.Choice(["pretrained", "scratch"]),
              default=TrainJointConfig.vocoder_init,
              help="release HiFi-GAN weights, or a seeded random generator", hidden=True)
@click.option("--critics_only", is_flag=True, default=TrainJointConfig.critics_only,
              help="train MPD and MSD only; decoder and vocoder stay frozen", hidden=True)
@click.option("--crop_frames", type=int, default=TrainJointConfig.crop_frames, hidden=True)
@click.option("--checkpointing_steps", type=int, default=TrainJointConfig.checkpointing_steps,
              help="save a checkpoint every this many steps")
@click.option("--logging_steps", type=int, default=TrainJointConfig.logging_steps,
              help="send training metrics to mlflow every this many steps", hidden=True)
@click.option("--init_from", default=TrainJointConfig.init_from,
              help="bundle whose weights seed the run; optimizer state restarts", hidden=True)
@click.option("--init_critics_from", default=TrainJointConfig.init_critics_from,
              help="bundle whose MPD and MSD seed the critics; generators stay as built", hidden=True)
@click.option("--data_workers", type=int, default=TrainJointConfig.data_workers, hidden=True)
@click.option("--data_prefetch", type=int, default=TrainJointConfig.data_prefetch, hidden=True)
@click.option("--precision", type=click.Choice(["bf16", "fp32", "fp16"]),
              default=TrainJointConfig.precision,
              help="fp32 trains without autocast and with TF32 off; fp16 scales each loss", hidden=True)
@click.option("--seed", type=int, default=TrainJointConfig.seed)
@click.option("--device", default=TrainJointConfig.device)
def train_joint(**option_dict):
    cfg = TrainJointConfig(**option_dict)
    apply_data_root(cfg.data_root)
    writer_cfg = TrainWriterConfig()
    writer = EvalResultWriter.make_from_dir(writer_cfg, cfg.out_dir)
    item_seed = CacheItemSeed(cfg.seed, writer_cfg.item_seed_offset)
    builder = TrainJointBuilder(cfg, writer, item_seed)
    trainer = builder.make_trainer()
    trainer.run()


if __name__ == "__main__":
    main()
