from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from torch import nn

from ..dataset.registry import VaeBatch, get_domain
from ..dataset.training_loader import DecoderLatentDataset
from ..evaluation.registry import EvalProtocol, FadProbeWindow
from ..metrics.scoring import MetricScoring
from ..models.stable_audio_open.registry import SpecVaeLoss, SpecVaeOptim
from ..runtime.dataroot import get_data_root
from .loss_function import DiscriminatorLossTerms, LossStftReconstructionModules


@dataclass(frozen=True)
class VaeOptimizers:
    decoder: torch.optim.Optimizer
    discriminator: torch.optim.Optimizer
    decoder_schedule: torch.optim.lr_scheduler.LRScheduler
    discriminator_schedule: torch.optim.lr_scheduler.LRScheduler


@dataclass(frozen=True)
class VaeTrainModules:
    decoder_model: nn.Module
    discriminator_model: nn.Module
    stft_reconstruction_model: LossStftReconstructionModules


@dataclass
class VaeLossValues:
    reconstructed_audio: torch.Tensor
    sum_difference: torch.Tensor
    channel: torch.Tensor
    stft_reconstruction: torch.Tensor
    adversarial: torch.Tensor
    feature_matching: torch.Tensor
    discriminator: torch.Tensor


class CkptBlobLoader:
    # Bundle contents as stored; a missing key fails at use.
    def load_state(self, blob: Path, device: str):
        blob_bytes = blob.read_bytes()
        buffer = io.BytesIO(blob_bytes)
        state = torch.load(buffer, map_location=device, weights_only=True)
        return state


class VaeOptimizerFactory:
    def __init__(self, optim: SpecVaeOptim):
        self.optim = optim

    # Warmup then linear decay, both counted from start.
    def make_lr_schedule(self, optimizer: torch.optim.Optimizer, start: int):
        optim = self.optim
        base = optimizer.param_groups[0]["lr"]
        end_factor = 0.0
        if base:
            end_factor = optim.final_lr / base
        decay_iters = max(1, optim.total_steps - start - optim.warmup_steps)
        decay = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=end_factor, total_iters=decay_iters,
        )
        stage_list = [decay]
        milestone_list = []
        if optim.warmup_steps >= 1:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0 / optim.warmup_steps, end_factor=1.0,
                total_iters=optim.warmup_steps,
            )
            stage_list.insert(0, warmup)
            milestone_list.insert(0, start + optim.warmup_steps)
        if start >= 1:
            # the lane idles until start; the held value never applies
            hold_factor = 1.0 / max(1, optim.warmup_steps)
            hold = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=hold_factor, total_iters=start,
            )
            stage_list.insert(0, hold)
            milestone_list.insert(0, start)
        schedule = decay
        if len(stage_list) > 1:
            schedule = torch.optim.lr_scheduler.SequentialLR(
                optimizer, stage_list, milestones=milestone_list,
            )
        return schedule

    # AdamW per net, each with its linear schedule.
    def make_optimizers(self, modules: VaeTrainModules, decoder_start: int):
        optim = self.optim
        decoder_param = modules.decoder_model.parameters()
        decoder_opt = torch.optim.AdamW(
            decoder_param, lr=optim.decoder_lr, betas=optim.betas,
            weight_decay=optim.weight_decay,
        )
        discriminator_param = modules.discriminator_model.parameters()
        discriminator_opt = torch.optim.AdamW(
            discriminator_param, lr=optim.discriminator_lr, betas=optim.betas,
            weight_decay=optim.weight_decay,
        )
        decoder_schedule = self.make_lr_schedule(decoder_opt, decoder_start)
        discriminator_schedule = self.make_lr_schedule(discriminator_opt, 0)
        optimizers = VaeOptimizers(
            decoder=decoder_opt,
            discriminator=discriminator_opt,
            decoder_schedule=decoder_schedule,
            discriminator_schedule=discriminator_schedule,
        )
        return optimizers

    # Weights, optimizer moments and schedules at one step.
    def make_ckpt_bundle(self, modules: VaeTrainModules, optimizers: VaeOptimizers, step: int):
        bundle = {
            "decoder": modules.decoder_model.state_dict(),
            "discriminator": modules.discriminator_model.state_dict(),
            "opt_decoder": optimizers.decoder.state_dict(),
            "opt_discriminator": optimizers.discriminator.state_dict(),
            "schedule_decoder": optimizers.decoder_schedule.state_dict(),
            "schedule_discriminator": optimizers.discriminator_schedule.state_dict(),
            "step": step,
        }
        return bundle


class VaeStepRunner:
    def __init__(self, modules: VaeTrainModules, loss_spec: SpecVaeLoss,
                 loss_terms: DiscriminatorLossTerms):
        self.modules = modules
        self.loss_spec = loss_spec
        self.loss_terms = loss_terms

    # Decoder audio and every weighted loss term for one batch.
    def make_loss_values(self, batch: VaeBatch):
        modules = self.modules
        loss_spec = self.loss_spec
        stft_model = modules.stft_reconstruction_model
        reconstructed_audio = modules.decoder_model(batch.latent)  # (B, C, S)
        # mid and side STFT distance over seven resolutions
        sum_term = stft_model.sum_difference(batch.ground_truth, reconstructed_audio)
        sum_difference = loss_spec.stft_reconstruction_weight * sum_term
        # left and right STFT distance over the same resolutions
        channel_term = stft_model.channel(batch.ground_truth, reconstructed_audio)
        channel = loss_spec.stft_reconstruction_weight * channel_term
        # release recipe adds both spectral terms at weight one
        stft_reconstruction = sum_difference + channel
        terms = self.loss_terms.compute_terms(
            modules.discriminator_model, batch.ground_truth, reconstructed_audio,
        )
        adversarial = loss_spec.adversarial_weight * terms.adversarial
        feature_matching = loss_spec.feature_weight * terms.feature_matching
        values = VaeLossValues(
            reconstructed_audio=reconstructed_audio,
            sum_difference=sum_difference,
            channel=channel,
            stft_reconstruction=stft_reconstruction,
            adversarial=adversarial,
            feature_matching=feature_matching,
            discriminator=terms.discriminator,
        )
        return values

    # Update the discriminator once over accumulated micro-batches.
    def train_discriminator(self, optimizers: VaeOptimizers, batch_tuple: tuple):
        modules = self.modules
        optimizers.decoder.zero_grad(set_to_none=True)
        optimizers.discriminator.zero_grad(set_to_none=True)
        modules.discriminator_model.requires_grad_(True)
        scale = 1.0 / len(batch_tuple)
        discriminator_loss_mean = 0.0
        for batch_item in batch_tuple:
            device_kind = batch_item.latent.device.type
            # fixed bf16 autocast; backends stay as codec load set
            with torch.autocast(device_type=device_kind, dtype=torch.bfloat16):
                # upstream keeps this attached; we skip the dead decoder graph
                with torch.no_grad():
                    reconstructed_audio = modules.decoder_model(batch_item.latent)  # (B, C, S)
                terms = self.loss_terms.compute_terms(
                    modules.discriminator_model, batch_item.ground_truth, reconstructed_audio,
                )
                discriminator_loss = terms.discriminator  # ()
            scaled_loss = scale * discriminator_loss  # ()
            scaled_loss.backward()
            detached = discriminator_loss.detach()  # ()
            discriminator_loss_mean += scale * float(detached)
        optimizers.discriminator.step()
        row = {
            "lane": "discriminator",
            "total": discriminator_loss_mean,
            "discriminator": discriminator_loss_mean,
        }
        return row

    # Update the decoder once over accumulated micro-batches.
    def train_decoder(self, optimizers: VaeOptimizers, batch_tuple: tuple):
        modules = self.modules
        optimizers.decoder.zero_grad(set_to_none=True)
        optimizers.discriminator.zero_grad(set_to_none=True)
        modules.discriminator_model.requires_grad_(False)
        scale = 1.0 / len(batch_tuple)
        row = {"lane": "decoder"}
        key_list = [
            "total", "stft_reconstruction", "sum_difference", "channel",
            "adversarial", "feature_matching",
        ]
        for key_item in key_list:
            row[key_item] = 0.0
        for batch_item in batch_tuple:
            device_kind = batch_item.latent.device.type
            with torch.autocast(device_type=device_kind, dtype=torch.bfloat16):
                values = self.make_loss_values(batch_item)
                total = values.stft_reconstruction  # ()
                # frozen discriminator still passes gradients to the decoder
                total = total + values.adversarial  # ()
                total = total + values.feature_matching  # ()
            scaled_total = scale * total  # ()
            scaled_total.backward()
            row["total"] += scale * float(total.detach())
            row["stft_reconstruction"] += scale * float(values.stft_reconstruction.detach())
            row["sum_difference"] += scale * float(values.sum_difference.detach())
            row["channel"] += scale * float(values.channel.detach())
            row["adversarial"] += scale * float(values.adversarial.detach())
            row["feature_matching"] += scale * float(values.feature_matching.detach())
        optimizers.decoder.step()
        return row

    # Every loss component of one held-out batch.
    def score_batch(self, batch: VaeBatch):
        device_kind = batch.latent.device.type
        with torch.autocast(device_type=device_kind, dtype=torch.bfloat16):
            values = self.make_loss_values(batch)
        total_tensor = values.stft_reconstruction + values.adversarial + values.feature_matching
        score_dict = {
            "total": float(total_tensor),
            "stft_reconstruction": float(values.stft_reconstruction),
            "sum_difference": float(values.sum_difference),
            "channel": float(values.channel),
            "adversarial": float(values.adversarial),
            "feature_matching": float(values.feature_matching),
            "discriminator": float(values.discriminator),
        }
        return score_dict

    # Hash-fixed crop onset per picked clip, so eval never moves.
    def get_onset_list(self, test_set: DecoderLatentDataset, pick_list: list, tracking,
                       crop_frames: int):
        onset_list = []
        for pick_item in pick_list:
            source = test_set.rows[pick_item].source
            source_hash = DecoderLatentDataset.get_source_hash(tracking, source, "onset")
            span = test_set.get_crop_span(pick_item, crop_frames)
            onset_list.append(source_hash % span)
        return onset_list

    # Per-domain and pooled means from clip-weighted sums.
    def make_holdout_tables(self, sum_dict: dict, count_dict: dict):
        table_dict = {}
        for domain_item, bucket in sum_dict.items():
            domain_table = {}
            for key_item, value_item in bucket.items():
                domain_table[key_item] = value_item / count_dict[domain_item]
            table_dict[domain_item] = domain_table
        count_values = count_dict.values()
        pooled = sum(count_values)
        bucket_list = list(sum_dict.values())
        first_bucket = bucket_list[0]
        all_table = {}
        for key_item in first_bucket:
            key_total = 0
            for domain_item in sum_dict:
                key_total = key_total + sum_dict[domain_item][key_item]
            all_table[key_item] = key_total / pooled
        table_dict["all"] = all_table
        return table_dict

    # Deterministic held-out losses per domain, gradient free.
    def run_holdout(self, codec, test_set: DecoderLatentDataset, tracking,
                    eval_batch_clips: int, crop_frames: int, device: str):
        modules = self.modules
        modules.decoder_model.eval()
        modules.discriminator_model.eval()
        sum_dict = {}
        count_dict = {}
        with torch.no_grad():
            pick_group_list = DecoderLatentDataset.collect_samples(test_set.rows, eval_batch_clips)
            for pick_list in pick_group_list:
                onset_list = self.get_onset_list(test_set, pick_list, tracking, crop_frames)
                latent, wav = test_set.load_crop_batch(pick_list, crop_frames, onset_list)
                host_batch = VaeBatch(latent=latent, ground_truth=wav)
                batch = DecoderLatentDataset.load_batch_device(host_batch, codec, device)
                score_dict = self.score_batch(batch)
                first_source = test_set.rows[pick_list[0]].source
                domain = get_domain(first_source)
                count_dict[domain] = count_dict.get(domain, 0) + len(pick_list)
                if domain not in sum_dict:
                    empty_bucket = {}
                    for key_item in score_dict:
                        empty_bucket[key_item] = 0.0
                    sum_dict[domain] = empty_bucket
                bucket = sum_dict[domain]
                for key_item, value_item in score_dict.items():
                    bucket[key_item] += value_item * len(pick_list)
        table_dict = self.make_holdout_tables(sum_dict, count_dict)
        modules.decoder_model.train()
        modules.discriminator_model.train()
        return table_dict


class FadProbeStore:
    def __init__(self, test_set: DecoderLatentDataset, decoded: list, cache_dir: Path,
                 window: FadProbeWindow):
        self.test_set = test_set
        self.decoded = decoded
        self.cache_dir = cache_dir
        self.window = window
        self.count = len(decoded)

    # Probe clips score at true length under the user-set window.
    def get_window(self, metric: str = ""):
        window = self.window
        return window

    # Stream one source's clips at a requested layout.
    def get_clips(self, source: str, sample_rate: int, channels: int, metric: str = ""):
        test_set = self.test_set
        native = test_set.header["sample_rate"]
        for row_idx, row_item in enumerate(test_set.rows):
            if source == "gen":
                clip = self.decoded[row_idx]  # (C, S)
            else:
                origin = test_set.get_row_origin(row_item)
                samples = test_set.get_row_samples(row_item)
                segment = test_set.audio.load_segment(
                    row_item.source, native, test_set.audio_channels, origin, samples,
                )
                wav = segment[0]  # (C, S)
                clip = test_set.apply_row_replay(wav, row_item)  # (C, S)
            clip = torchaudio.functional.resample(clip, native, sample_rate)  # (C, S)
            if clip.shape[0] != channels:
                mono = clip.mean(dim=0, keepdim=True)  # (1, S)
                clip = mono.repeat(channels, 1)  # (C, S)
            yield clip


class FadProbe:
    def __init__(self, window: FadProbeWindow, device: str):
        self.window = window
        self.device = device

    # Row indices grouped by latent shape, so batches stack.
    def get_shape_groups(self, test_set: DecoderLatentDataset):
        group_dict = {}
        for row_idx, row_item in enumerate(test_set.rows):
            shard = test_set.get_shard_tensor(row_item)
            shape = tuple(shard[row_item.row].shape)
            if shape not in group_dict:
                group_dict[shape] = []
            group_dict[shape].append(row_idx)
        return group_dict

    # Decode every held-out latent with the training decoder.
    def make_decoded_clips(self, modules: VaeTrainModules, codec, test_set: DecoderLatentDataset):
        modules.decoder_model.eval()
        decoded = []
        for row_item in test_set.rows:
            decoded.append(None)
        group_dict = self.get_shape_groups(test_set)
        with torch.no_grad():
            for member_list in group_dict.values():
                for start in range(0, len(member_list), 8):
                    pick_list = member_list[start:start + 8]
                    latent_list = []
                    for pick_item in pick_list:
                        row = test_set.rows[pick_item]
                        shard = test_set.get_shard_tensor(row)
                        latent_list.append(shard[row.row])
                    stacked = torch.stack(latent_list)  # (B, D, T)
                    stacked = stacked.float()  # (B, D, T)
                    latents = stacked.to(self.device)  # (B, D, T)
                    latents = codec.apply_latent_scale(latents, "denormalize")  # (B, D, T)
                    with torch.autocast(device_type=latents.device.type, dtype=torch.bfloat16):
                        reconstructed_audio = modules.decoder_model(latents)  # (B, C, S)
                    for pick_item, clip_item in zip(pick_list, reconstructed_audio):
                        clamped = clip_item.clamp(-1, 1)  # (C, S)
                        clamped = clamped.float()  # (C, S)
                        decoded[pick_item] = clamped.cpu()
        modules.decoder_model.train()
        return decoded

    # Scorer over the bundled protocol's metric suite.
    def make_scoring(self):
        eval_protocol = EvalProtocol.make_from_dir("")
        data_root = get_data_root()
        ckpt_dir = data_root / "fad_checkpoints"
        scoring = MetricScoring.make_from_suite(eval_protocol.metric_suite, ckpt_dir, self.device)
        return scoring

    # Drop metric rows built from the previous decoder weights.
    def clear_cache(self, cache_dir: Path):
        for pattern_item in ("*.npy", "*.npz"):
            for path_item in cache_dir.glob(pattern_item):
                if "ref" not in path_item.stem:
                    path_item.unlink()

    # Score held-out decodes with the exact offline metrics.
    def run(self, modules: VaeTrainModules, codec, test_set: DecoderLatentDataset,
            name_list, pair_list, cache_dir: Path):
        decoded = self.make_decoded_clips(modules, codec, test_set)
        # reference embeddings persist; decoded embeddings go stale each pass
        self.clear_cache(cache_dir)
        store = FadProbeStore(test_set, decoded, cache_dir, self.window)
        scoring = self.make_scoring()
        score = scoring.get_metrics(store, name_list, pair_list)
        flat_dict = {}
        for name_item, table in score.metric_dict.items():
            for key_item, value_item in table.items():
                flat_dict[f"{name_item}/{key_item}"] = float(value_item)
        return flat_dict
