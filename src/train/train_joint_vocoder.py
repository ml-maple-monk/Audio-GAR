from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import torch
from torch import nn

from ..dataset.registry import VaeBatch
from ..models.tango2.registry import SpecJointScopes, SpecJointWindow
from ..models.tango2.tokenizer import MelDecoderModule
from ..models.tango2.vendor.tokenizer_arch import HIFIGAN_16K_64, AttrDict, Generator
from ..models.tango_music.tokenizer import MelDecoderModule as MusicMelDecoderModule
from ..models.tango_music.tokenizer import TangoMusicTokenizer
from .loss_function import JointGeneratorLoss, VocoderLossSet


@dataclass(frozen=True)
class VocoderJointOptimizers:
    generator: torch.optim.Optimizer | None
    vocoder_discriminator: torch.optim.Optimizer
    generator_schedule: torch.optim.lr_scheduler.LRScheduler | None
    critic_schedule: torch.optim.lr_scheduler.LRScheduler


@dataclass
class ContextWindow:
    mel_cut: torch.Tensor
    real: torch.Tensor
    trim: int


@dataclass
class JointWindow:
    mel_cut: torch.Tensor
    batch: VaeBatch


@dataclass
class JointWindowResult:
    window_list: list
    given_list: list
    skipped: int
    stat_dict: dict


@dataclass
class CriticLaneResult:
    wave_list: list
    critic_loss: float
    stat_dict: dict


@dataclass
class GeneratorLaneResult:
    sum_dict: dict
    scaler: torch.amp.GradScaler | None


class MelDecoderFactory:
    # Family decoder module over the codec's frozen VAE.
    def make_decoder(self, codec):
        if isinstance(codec, TangoMusicTokenizer):
            decoder = MusicMelDecoderModule(codec.vae, codec.spec)
        else:
            decoder = MelDecoderModule(codec.vae, codec.spec)
        return decoder


class VocoderTrainModules(nn.Module):
    def __init__(self, generator_model, mpd_model, msd_model, decoder_model):
        super().__init__()
        self.generator_model = generator_model
        self.mpd_model = mpd_model
        self.msd_model = msd_model
        self.decoder_model = decoder_model


class VocoderGeneratorFactory:
    # Load fused release weights into weight-norm parameters.
    def apply_fused_weight_norm(self, generator: nn.Module, fused: dict):
        normed = generator.state_dict()
        for name_item, value_item in fused.items():
            if f"{name_item}_v" in normed:
                normed[f"{name_item}_v"] = value_item.clone()
                normed[f"{name_item}_g"] = torch.norm_except_dim(value_item, 2, 0)
            else:
                normed[name_item] = value_item.clone()
        generator.load_state_dict(normed, strict=True)

    # Weight-normed generator carrying the release weights.
    def make_release_generator(self, codec):
        config = AttrDict(HIFIGAN_16K_64)
        generator = Generator(config)
        fused = codec.vae.vocoder.state_dict()
        self.apply_fused_weight_norm(generator, fused)
        return generator

    # Weight-normed generator at its random init; seed it first.
    def make_scratch_generator(self):
        config = AttrDict(HIFIGAN_16K_64)
        generator = Generator(config)
        return generator


class VocoderJointModules(nn.Module):
    def __init__(self, decoder_model, vocoder_modules, vocoder_loss_weight: float = 1.0):
        super().__init__()
        self.decoder_model = decoder_model
        self.vocoder_modules = vocoder_modules
        self.vocoder_loss_weight = vocoder_loss_weight
        vocoder_modules.mpd_model.requires_grad_(True)
        vocoder_modules.mpd_model.train()
        vocoder_modules.msd_model.requires_grad_(True)
        vocoder_modules.msd_model.train()


class JointDecoder:
    def __init__(self, modules: VocoderJointModules):
        self.modules = modules

    # Mel prediction for one latent batch.
    def decode_latent(self, latent: torch.Tensor):
        mel_pred = self.modules.decoder_model(latent)  # (B, 1, T, M)
        return mel_pred


@dataclass
class JointStepFns:
    decode: object
    loss_set: VocoderLossSet
    compiled: bool
    autocast: bool
    autocast_dtype: torch.dtype
    scaler: torch.amp.GradScaler | None
    scaler_generator: torch.amp.GradScaler | None

    # Forbid recompiles once warmup has built every graph.
    def apply_compile_fence(self, step: int, fence_step: int):
        if self.compiled and step == fence_step:
            torch.compiler.set_stance("fail_on_recompile")


class JointStepFnFactory:
    # Step callables; cuda compiles the decode.
    def make_step_fns(self, modules: VocoderJointModules, device: str, loss_set: VocoderLossSet,
                      autocast: bool, autocast_dtype: torch.dtype):
        joint_decoder = JointDecoder(modules)
        decode = joint_decoder.decode_latent
        compiled = device.startswith("cuda")
        if compiled:
            decode = torch.compile(joint_decoder.decode_latent)
        fp16 = autocast and autocast_dtype == torch.float16
        scaler = None
        scaler_generator = None
        if fp16:
            scaler = torch.amp.GradScaler("cuda")
            scaler_generator = torch.amp.GradScaler("cuda")
        fns = JointStepFns(
            decode=decode,
            loss_set=loss_set,
            compiled=compiled,
            autocast=autocast,
            autocast_dtype=autocast_dtype,
            scaler=scaler,
            scaler_generator=scaler_generator,
        )
        return fns


class JointScopeFreezer:
    def __init__(self, scopes: SpecJointScopes):
        self.scopes = scopes

    # Prefixes a scope releases; none releases nothing.
    def get_prefix_tuple(self, scope: str, named_tuple: tuple):
        prefix_tuple = named_tuple
        if scope == "all":
            prefix_tuple = ()
        if scope == "none":
            prefix_tuple = None
        return prefix_tuple

    # Freeze the net, then release the scope's weights.
    def freeze_scope(self, model: nn.Module, prefix_tuple: tuple | None):
        model.requires_grad_(False)
        model.train()
        released_list = []
        if prefix_tuple is not None:
            for name_item, value_item in model.named_parameters():
                if not prefix_tuple or name_item.startswith(prefix_tuple):
                    value_item.requires_grad_(True)
                    released_list.append(name_item)
        return released_list

    # Release the decoder and vocoder weights each scope names.
    def apply_scopes(self, modules: VocoderJointModules, decoder_scope: str, vocoder_scope: str):
        decoder_prefix = self.get_prefix_tuple(decoder_scope, self.scopes.decoder_first)
        vocoder_prefix = self.get_prefix_tuple(vocoder_scope, self.scopes.vocoder_last)
        generator_model = modules.vocoder_modules.generator_model
        trainable_dict = {}
        trainable_dict["decoder"] = self.freeze_scope(modules.decoder_model, decoder_prefix)
        trainable_dict["vocoder"] = self.freeze_scope(generator_model, vocoder_prefix)
        return trainable_dict


class JointWeightLoader:
    # Restore every net's weights from one bundle.
    def apply_weights(self, modules: VocoderJointModules, state: dict):
        wave = modules.vocoder_modules
        vocoder_state = state["vocoder"]
        modules.decoder_model.load_state_dict(state["decoder"], strict=True)
        wave.generator_model.load_state_dict(vocoder_state["generator"], strict=True)
        wave.mpd_model.load_state_dict(vocoder_state["mpd"], strict=True)
        wave.msd_model.load_state_dict(vocoder_state["msd"], strict=True)


class JointOptimizerFactory:
    def __init__(self, optim, vocoder_optim, steps: int):
        self.optim = optim
        self.vocoder_optim = vocoder_optim
        self.steps = steps

    # Parameters still asking for gradients, in definition order.
    def get_trainable(self, named_iter):
        param_list = []
        for name_item, value_item in named_iter:
            if value_item.requires_grad:
                param_list.append(value_item)
        return param_list

    # Torch LinearLR from the base rate down to zero.
    def make_linear_schedule(self, optimizer: torch.optim.Optimizer):
        schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.steps,
        )
        return schedule

    # One AdamW over both generators, one over the critics.
    def make_optimizers(self, modules: VocoderJointModules):
        wave = modules.vocoder_modules
        vocoder_optim = self.vocoder_optim
        decoder_named = modules.decoder_model.named_parameters()
        generator_named = wave.generator_model.named_parameters()
        generator_chain = chain(decoder_named, generator_named)
        generator_param_list = self.get_trainable(generator_chain)
        generator_opt = None
        if generator_param_list:
            generator_lr = torch.tensor(self.optim.lr)
            generator_opt = torch.optim.AdamW(
                generator_param_list, lr=generator_lr, betas=vocoder_optim.betas,
                weight_decay=vocoder_optim.weight_decay,
            )
        mpd_named = wave.mpd_model.named_parameters()
        msd_named = wave.msd_model.named_parameters()
        critic_chain = chain(mpd_named, msd_named)
        critic_param_list = self.get_trainable(critic_chain)
        critic_lr = torch.tensor(vocoder_optim.lr)
        critic_opt = torch.optim.AdamW(
            critic_param_list, lr=critic_lr, betas=vocoder_optim.betas,
            weight_decay=vocoder_optim.weight_decay,
        )
        generator_schedule = None
        if generator_opt:
            generator_schedule = self.make_linear_schedule(generator_opt)
        critic_schedule = self.make_linear_schedule(critic_opt)
        optimizers = VocoderJointOptimizers(
            generator=generator_opt,
            vocoder_discriminator=critic_opt,
            generator_schedule=generator_schedule,
            critic_schedule=critic_schedule,
        )
        return optimizers

    # Weights, moments and schedules at one step.
    def make_ckpt_bundle(self, modules: VocoderJointModules, optimizers: VocoderJointOptimizers,
                         step: int):
        wave = modules.vocoder_modules
        generator_state = None
        if optimizers.generator:
            generator_state = optimizers.generator.state_dict()
        generator_schedule_state = None
        if optimizers.generator_schedule:
            generator_schedule_state = optimizers.generator_schedule.state_dict()
        bundle = {
            "decoder": modules.decoder_model.state_dict(),
            "vocoder": {
                "generator": wave.generator_model.state_dict(),
                "mpd": wave.mpd_model.state_dict(),
                "msd": wave.msd_model.state_dict(),
            },
            "opt_generator": generator_state,
            "opt_critic": optimizers.vocoder_discriminator.state_dict(),
            "sched_generator": generator_schedule_state,
            "sched_critic": optimizers.critic_schedule.state_dict(),
            "step": step,
        }
        return bundle


class GradNormReader:
    # Total grad norm over an optimizer; zero when no grad.
    def get_optimizer_norm(self, optimizer: torch.optim.Optimizer):
        grad_list = []
        for group_item in optimizer.param_groups:
            for param_item in group_item["params"]:
                if param_item.grad is not None:
                    grad_list.append(param_item.grad)
        norm = 0.0
        if grad_list:
            total = torch.nn.utils.get_total_norm(grad_list)  # ()
            norm = float(total)
        return norm

    # Total grad norm of one module; zero when no grad.
    def get_module_norm(self, module: nn.Module):
        grad_list = []
        for param_item in module.parameters():
            if param_item.grad is not None:
                grad_list.append(param_item.grad)
        norm = 0.0
        if grad_list:
            total = torch.nn.utils.get_total_norm(grad_list)  # ()
            norm = float(total)
        return norm


class JointWindowCutter:
    def __init__(self, window_spec: SpecJointWindow):
        self.window_spec = window_spec

    # Cached latents carry rare NaN; one poisons every later step.
    def check_finite_batch(self, batch: VaeBatch):
        latent_ok = torch.isfinite(batch.latent).all()
        wav_ok = torch.isfinite(batch.ground_truth).all()
        finite = bool(latent_ok and wav_ok)
        return finite

    # Seeded onsets that leave vocoder context on both sides.
    def make_window_onsets(self, latent_frames: int, frames: int, total: int, seed: int):
        margin = self.window_spec.margin_frames
        room = latent_frames - frames - 2 * margin
        generator = torch.Generator()
        generator.manual_seed(seed)
        draw = torch.randint(0, room + 1, (total,), generator=generator)  # (N,)
        shifted = draw + margin  # (N,)
        onset_list = shifted.tolist()
        return onset_list

    # Mel windows with context, and the exact real segments.
    def make_context_window(self, mel_pred: torch.Tensor, batch: VaeBatch, frames: int,
                            count: int, seed: int):
        latent_frames = batch.latent.shape[-1]
        margin = self.window_spec.margin_frames
        mel_step = mel_pred.shape[2] // latent_frames
        wav_step = batch.ground_truth.shape[-1] // latent_frames
        clip_list = []
        for clip_idx in range(mel_pred.shape[0]):
            for repeat_idx in range(count):
                clip_list.append(clip_idx)
        onset_list = self.make_window_onsets(latent_frames, frames, len(clip_list), seed)
        mel_list = []
        real_list = []
        for clip_idx, start in zip(clip_list, onset_list):
            mel_start = (start - margin) * mel_step
            mel_stop = (start + frames + margin) * mel_step
            mel_list.append(mel_pred[clip_idx, :, mel_start:mel_stop])
            wav_start = start * wav_step
            wav_stop = (start + frames) * wav_step
            real_list.append(batch.ground_truth[clip_idx, :, wav_start:wav_stop])
        mel_cut = torch.stack(mel_list)  # (N, 1, Tw, M)
        real = torch.stack(real_list)  # (N, 1, Sw)
        window = ContextWindow(mel_cut=mel_cut, real=real, trim=margin * wav_step)
        return window

    # One DC and peak stage per window, either side.
    def apply_window_norm(self, wav: torch.Tensor):
        wav = wav.float()  # (N, 1, S)
        dc = wav.mean(dim=(1, 2), keepdim=True)  # (N, 1, 1)
        centered = wav - dc  # (N, 1, S)
        magnitude = centered.abs()  # (N, 1, S)
        peak = magnitude.amax(dim=(1, 2), keepdim=True)  # (N, 1, 1)
        divided = centered / (peak + 1e-8)  # (N, 1, S)
        normed = divided * self.window_spec.peak  # (N, 1, S)
        return normed

    # Level statistics the critics could tell the sides by.
    def make_window_stats(self, real: torch.Tensor, fake: torch.Tensor):
        stat_dict = {}
        real_detached = real.detach()  # (N, 1, S)
        fake_detached = fake.detach()  # (N, 1, S)
        for side_item, wav in (("real", real_detached), ("fake", fake_detached)):
            power = wav.pow(2)  # (N, 1, S)
            mean_power = power.mean(dim=(1, 2))  # (N,)
            rms = mean_power.sqrt()  # (N,)
            stat_dict[f"window_rms_{side_item}"] = float(rms.mean())
            magnitude = wav.abs()  # (N, 1, S)
            peak = magnitude.amax(dim=(1, 2))  # (N,)
            stat_dict[f"window_peak_{side_item}"] = float(peak.mean())
            dc = wav.mean(dim=(1, 2))  # (N,)
            dc_size = dc.abs()  # (N,)
            stat_dict[f"window_dc_{side_item}"] = float(dc_size.mean())
            crest = peak / (rms + 1e-8)  # (N,)
            stat_dict[f"window_crest_{side_item}"] = float(crest.mean())
        return stat_dict

    # Vocode context windows in fp32; both sides share one norm.
    def make_windows(self, modules: VocoderJointModules, fns: JointStepFns, batch_tuple: tuple,
                     frames: int, count: int, seed: int, grad: bool):
        window_list = []
        given_list = []
        skipped = 0
        stat_dict = {"window_silent": 0}
        generator_model = modules.vocoder_modules.generator_model
        for batch_item in batch_tuple:
            if self.check_finite_batch(batch_item):
                kind = batch_item.latent.device.type
                with torch.set_grad_enabled(grad), torch.autocast(
                    device_type=kind, dtype=fns.autocast_dtype, enabled=fns.autocast,
                ):
                    mel_pred = fns.decode(batch_item.latent)  # (B, 1, T, M)
                context = self.make_context_window(mel_pred, batch_item, frames, count, seed)
                # a silent real window would normalise to noise
                real_magnitude = context.real.abs()  # (N, 1, Sw)
                real_peak = real_magnitude.amax(dim=(1, 2))  # (N,)
                loud = real_peak >= self.window_spec.silence_peak  # (N,)
                quiet = ~loud  # (N,)
                stat_dict["window_silent"] += int(quiet.sum())
                if bool(loud.any()):
                    mel_cut = context.mel_cut[loud]  # (L, 1, Tw, M)
                    real = context.real[loud]  # (L, 1, Sw)
                    # bf16 vocoding left 19-26 dB noise on fakes
                    with torch.set_grad_enabled(grad), torch.autocast(device_type=kind, enabled=False):
                        mel_float = mel_cut.float()  # (L, 1, Tw, M)
                        mel_squeezed = mel_float.squeeze(1)  # (L, Tw, M)
                        mel_input = mel_squeezed.permute(0, 2, 1)  # (L, M, Tw)
                        wav = generator_model(mel_input)  # (L, 1, Sv)
                    trim = context.trim
                    wav = wav[..., trim:trim + real.shape[-1]]  # (L, 1, Sw)
                    real = self.apply_window_norm(real)  # (L, 1, Sw)
                    fake = self.apply_window_norm(wav)  # (L, 1, Sw)
                    window_stat_dict = self.make_window_stats(real, fake)
                    stat_dict.update(window_stat_dict)
                    window_batch = VaeBatch(latent=batch_item.latent, ground_truth=real)
                    window_list.append(JointWindow(mel_cut=mel_cut, batch=window_batch))
                    given_list.append(fake)
                else:
                    skipped += 1
            else:
                skipped += 1
        result = JointWindowResult(
            window_list=window_list, given_list=given_list, skipped=skipped, stat_dict=stat_dict,
        )
        return result


class JointStepRunner:
    def __init__(self, modules: VocoderJointModules, fns: JointStepFns,
                 window_cutter: JointWindowCutter, grad_reader: GradNormReader,
                 window_spec: SpecJointWindow, generator_loss: JointGeneratorLoss):
        self.modules = modules
        self.fns = fns
        self.window_cutter = window_cutter
        self.grad_reader = grad_reader
        self.window_spec = window_spec
        self.generator_loss = generator_loss

    # Fit detached waveform critics; keep the attached fakes.
    def run_critic_lane(self, optimizers: VocoderJointOptimizers, window_list: list,
                        given_list: list, scale: float):
        fns = self.fns
        critics = self.modules.vocoder_modules
        optimizer = optimizers.vocoder_discriminator
        scaler = fns.scaler
        optimizer.zero_grad(set_to_none=True)
        wave_list = []
        critic_sum = 0.0
        for window_idx, window_item in enumerate(window_list):
            batch = window_item.batch
            kind = batch.latent.device.type
            with torch.autocast(device_type=kind, dtype=fns.autocast_dtype, enabled=fns.autocast):
                # generated audio arrives whole-crop normalised like the real loader
                wav_pred = given_list[window_idx]  # (L, 1, Sw)
                with torch.set_grad_enabled(True):
                    detached = wav_pred.detach()  # (L, 1, Sw)
                    critic_loss = fns.loss_set.make_critic_loss(batch.ground_truth, detached)  # ()
            wave_list.append(wav_pred)
            loss = scale * critic_loss  # ()
            if scaler is not None:
                loss = scaler.scale(loss)  # ()
            loss.backward()
            critic_sum += scale * float(critic_loss.detach())
        stat_dict = {}
        if window_list:
            if scaler is not None:
                # unscaled grads feed the monitor and the inf check
                scaler.unscale_(optimizer)
            stat_dict["grad_norm_mpd"] = self.grad_reader.get_module_norm(critics.mpd_model)
            stat_dict["grad_norm_msd"] = self.grad_reader.get_module_norm(critics.msd_model)
            if scaler is not None:
                before = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                after = float(scaler.get_scale())
                stat_dict["amp_scale"] = after
                stat_dict["amp_step_skipped"] = float(after < before)
            else:
                optimizer.step()
        result = CriticLaneResult(wave_list=wave_list, critic_loss=critic_sum, stat_dict=stat_dict)
        return result

    # Backward the waveform losses into vocoder and decoder.
    def run_generator_lane(self, optimizers: VocoderJointOptimizers, window_list: list,
                           wave_list: list, scale: float):
        train = optimizers.generator is not None
        scaler = None
        if train:
            optimizers.generator.zero_grad(set_to_none=True)
            scaler = self.fns.scaler_generator
        sum_dict = {"total": 0.0}
        for window_item, wav_pred in zip(window_list, wave_list):
            with torch.set_grad_enabled(train):
                loss = self.generator_loss.make_loss(window_item.batch, wav_pred)
            total = self.modules.vocoder_loss_weight * loss.total  # ()
            if train:
                scaled = scale * total  # ()
                if scaler is not None:
                    scaled = scaler.scale(scaled)  # ()
                scaled.backward()
            sum_dict["total"] += scale * float(total.detach())
            for key_item, value_item in loss.value_dict.items():
                sum_dict[key_item] = sum_dict.get(key_item, 0.0) + scale * value_item
        if window_list and scaler is not None:
            scaler.unscale_(optimizers.generator)
        result = GeneratorLaneResult(sum_dict=sum_dict, scaler=scaler)
        return result

    # Plain step in bf16 or fp32; scaler-guarded step in fp16.
    def apply_generator_step(self, optimizer: torch.optim.Optimizer, scaler):
        stat_dict = {}
        if scaler is None:
            optimizer.step()
        else:
            before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            after = float(scaler.get_scale())
            stat_dict["amp_scale_generator"] = after
            stat_dict["amp_step_skipped_generator"] = float(after < before)
        return stat_dict

    # Generator grad norms on the cadence step only.
    def get_norm_dict(self, optimizers: VocoderJointOptimizers, step: int):
        norm_dict = {}
        if (step + 1) % self.window_spec.grad_norm_every == 0:
            generator_model = self.modules.vocoder_modules.generator_model
            norm_dict["grad_norm_generator"] = self.grad_reader.get_optimizer_norm(optimizers.generator)
            norm_dict["grad_norm_decoder"] = self.grad_reader.get_module_norm(self.modules.decoder_model)
            norm_dict["grad_norm_vocoder"] = self.grad_reader.get_module_norm(generator_model)
        return norm_dict

    # Critics first, then decoder and vocoder on one backward.
    def run_step(self, optimizers: VocoderJointOptimizers, batch_tuple: tuple, step: int,
                 vocoder_window: int, windows_per_clip: int, window_seed: int):
        scale = 1.0 / len(batch_tuple)
        train = optimizers.generator is not None
        windows = self.window_cutter.make_windows(
            self.modules, self.fns, batch_tuple, vocoder_window, windows_per_clip, window_seed, train,
        )
        critic = self.run_critic_lane(optimizers, windows.window_list, windows.given_list, scale)
        generator = self.run_generator_lane(optimizers, windows.window_list, critic.wave_list, scale)
        norm_dict = {}
        rate = 0.0
        if train:
            norm_dict = self.get_norm_dict(optimizers, step)
            if windows.window_list:
                step_dict = self.apply_generator_step(optimizers.generator, generator.scaler)
                norm_dict.update(step_dict)
            else:
                optimizers.generator.zero_grad(set_to_none=True)
            rate = float(optimizers.generator.param_groups[0]["lr"])
        critic_rate = float(optimizers.vocoder_discriminator.param_groups[0]["lr"])
        if train:
            optimizers.generator_schedule.step()
        optimizers.critic_schedule.step()
        row = {
            "lane": "joint_vocoder",
            "vocoder_discriminator": critic.critic_loss,
            "lr": rate,
            "critic_lr": critic_rate,
            "skipped": windows.skipped,
        }
        row.update(generator.sum_dict)
        row.update(norm_dict)
        row.update(critic.stat_dict)
        row.update(windows.stat_dict)
        return row
