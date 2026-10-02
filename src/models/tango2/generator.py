from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import click
import torch
from diffusers import DDPMScheduler
from torch import nn

from ...dataset.registry import get_cosine_noise_coefficients
from ...runtime.dataroot import apply_data_root, get_data_root
from . import registry
from .tokenizer import Tango2Tokenizer, apply_latent_fold
from .vendor.generator_arch import AudioDiffusion


# Text embeddings and mask, guidance branches concatenated.
@dataclass
class Conditioning:
    prompt_embeds: torch.Tensor
    prompt_mask: torch.Tensor
    cfg_scale: float


# Tango 2 text-to-audio latent generator.
class Tango2Generator(nn.Module):
    def __init__(self, name: str = "tango2-full", ckpt_path: str = "", device: str = "cuda",
                 determinism: bool = False, compile_blocks: bool = False):
        super().__init__()
        self.name = name

        self.spec = registry.spec(name)
        if ckpt_path:
            ckpt = ckpt_path
        else:
            ckpt = registry.get_ckpt_path(name)
        main_ckpt, text_encoder_path = self.make_main_source(ckpt, name)
        unet_kwargs = asdict(self.spec.unet)
        self.model = AudioDiffusion(text_encoder_path, unet_kwargs)

        if main_ckpt:
            state = torch.load(main_ckpt, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=True)

        self.model.to(device)
        self.model.eval()

        first_param = next(self.model.parameters())
        self.device = first_param.device
        first_unet_param = next(self.model.unet.parameters())
        self.dtype = first_unet_param.dtype
        self.scheduler = self.make_scheduler()
        self.codec = None
        if determinism:
            self.apply_determinism()
        else:
            self.apply_precision_policy()
        if compile_blocks:
            self.apply_compile()

    # Bundle manifest, or a direct ckpt and text encoder.
    def make_main_source(self, ckpt: str, name: str):
        if ckpt and ckpt.endswith(".json"):
            bundle = registry.load_bundle(ckpt)
            root = get_data_root()
            main_path = str(root / bundle.main)
            text_encoder_path = str(root / bundle.text_encoder)
        else:
            main_path = ckpt
            text_encoder_path = registry.get_text_encoder_path(name)
        return main_path, text_encoder_path

    # Upstream inference grid entries at or below start.
    def make_grid_suffix(self, scheduler, start: int, num_steps: int):
        scheduler.set_timesteps(num_steps)
        suffix = []
        for step_item in scheduler.timesteps:
            step_value = int(step_item)
            if step_value <= start:
                suffix.append(step_value)
        if not suffix:
            last_step = scheduler.timesteps[-1]
            suffix = [int(last_step)]
        return suffix

    # Scheduler on the schedule the checkpoint trained under.
    def make_scheduler(self):
        scheduler_kwargs = asdict(self.spec.scheduler)
        scheduler = DDPMScheduler(**scheduler_kwargs)
        return scheduler

    # Tf32 and autotuning, which determinism forbids.
    def apply_precision_policy(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Compile every U-Net stage independently.
    def apply_compile(self):
        unet = self.model.unet
        n_down = len(unet.down_blocks)
        for block_idx in range(n_down):
            block = unet.down_blocks[block_idx]
            unet.down_blocks[block_idx] = torch.compile(block, dynamic=False)
        unet.mid_block = torch.compile(unet.mid_block, dynamic=False)
        n_up = len(unet.up_blocks)
        for block_idx in range(n_up):
            block = unet.up_blocks[block_idx]
            unet.up_blocks[block_idx] = torch.compile(block, dynamic=False)

    # Tokenizer built on first decode, so generation stays light.
    def get_codec(self):
        if self.codec is None:
            device_name = str(self.device)
            self.codec = Tango2Tokenizer(self.name, device=device_name)
        return self.codec

    # Decode from the folded, scaled latent space.
    def decode_latent(self, latent: torch.Tensor):
        codec = self.get_codec()
        wav = codec.decode_latent(latent)
        return wav

    # Deterministic kernels, no tf32, no autotuning.
    def apply_determinism(self):
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Model-space latent shape for one batch.
    def get_latent_shape(self, batch_size: int):
        shape = (
            batch_size,
            self.spec.latent_channels,
            self.spec.generation_frames,
            self.spec.vae.mel_bins,
        )
        return shape

    # One independent noise draw per row, from that row's seed.
    def sample_noise_vector(self, batch_size: int, seeds: list[int]):
        if len(seeds) != batch_size:
            raise ValueError(f"{batch_size} rows need {batch_size} seeds, got {len(seeds)}")
        shape = self.get_latent_shape(1)
        row_list = []
        for seed_item in seeds:
            stream = torch.Generator(device=self.device)
            seed_value = int(seed_item)
            stream.manual_seed(seed_value)
            row = torch.randn(shape, generator=stream, device=self.device)
            row_list.append(row)
        noise = torch.cat(row_list)  # (B, C, F, bins)
        return noise

    # Text embeddings for the guided and null branches.
    @torch.no_grad()
    def make_conditioning(
        self,
        text_prompt: str | list[str],
        seconds_start: float | list[float] = 0.0,
        seconds_total: float | list[float] = 10.24,
        negative_prompt: str | None = None,
    ):
        if isinstance(text_prompt, str):
            prompts = [text_prompt]
        else:
            prompts = list(text_prompt)
        embeds, mask = self.model.encode_text_classifier_free(prompts, 1)
        cond = Conditioning(embeds, mask, self.spec.sampling.cfg_scale)
        return cond

    # Nearest schedule step matching these cosine coefficients.
    def get_timestep(self, coeff_a: torch.Tensor, coeff_b: torch.Tensor):
        norm_sq = coeff_a**2 + coeff_b**2
        norm = norm_sq.sqrt()
        ratio = coeff_a / norm
        wanted = float(ratio)
        alphas_sqrt = self.scheduler.alphas_cumprod.sqrt()
        gap = alphas_sqrt - wanted
        distance = gap.abs()
        nearest = distance.argmin()
        timestep = int(nearest)
        return timestep

    # One guided denoiser call over the doubled batch.
    def get_prediction(self, latent: torch.Tensor, step, cond: Conditioning):
        doubled = torch.cat([latent] * 2)
        doubled = self.scheduler.scale_model_input(doubled, step)
        unet_output = self.model.unet(
            doubled, step,
            encoder_hidden_states=cond.prompt_embeds,
            encoder_attention_mask=cond.prompt_mask,
        )
        predicted = unet_output.sample  # (2B, C, F, bins)
        null, guided = predicted.chunk(2)
        guidance = guided - null
        guided_pred = null + cond.cfg_scale * guidance
        return guided_pred

    # V-prediction resolved to the clean-sample estimate.
    def get_x0_estimate(self, z_t: torch.Tensor, alpha: torch.Tensor, predicted: torch.Tensor):
        signal_coeff = alpha.sqrt()
        signal = signal_coeff * z_t
        noise_var = 1.0 - alpha
        noise_coeff = noise_var.sqrt()
        noise = noise_coeff * predicted
        x0 = signal - noise
        return x0

    # Hand-rolled DDPM step; the library lacks per-row noise.
    def denoise_posterior_step(
        self, z_t: torch.Tensor, predicted: torch.Tensor, step, prev_step,
        variance_noise: torch.Tensor,
    ):
        step_idx = int(step)
        prev_idx = int(prev_step)
        alpha_t = self.scheduler.alphas_cumprod[step_idx]
        if prev_idx >= 0:
            alpha_prev = self.scheduler.alphas_cumprod[prev_idx]
        else:
            alpha_prev = self.scheduler.one
        beta_t = 1.0 - alpha_t
        beta_prev = 1.0 - alpha_prev
        current_alpha = alpha_t / alpha_prev
        current_beta = 1.0 - current_alpha
        x0 = self.get_x0_estimate(z_t, alpha_t, predicted)
        x0_coeff = alpha_prev.sqrt() * current_beta / beta_t
        z_coeff = current_alpha.sqrt() * beta_prev / beta_t
        mean = x0_coeff * x0 + z_coeff * z_t
        if step_idx <= 0:
            z_prev = mean
        else:
            variance_raw = (1.0 - alpha_prev) / beta_t * current_beta
            variance = variance_raw.clamp(min=1e-20)
            std = variance.sqrt()
            z_prev = mean + std * variance_noise
        return z_prev

    # Ancestral DDPM loop; every row draws its own variance noise.
    @torch.no_grad()
    def denoise_diffusion_loop(
        self, noise: torch.Tensor, cond: Conditioning, steps: int, seeds: list[int]
    ):
        self.scheduler.set_timesteps(steps, device=self.device)
        latent = noise * self.scheduler.init_noise_sigma
        stream_list = []
        for seed_item in seeds:
            stream = torch.Generator(device=self.device)
            seed_value = int(seed_item) + 1
            stream.manual_seed(seed_value)
            stream_list.append(stream)
        timesteps = self.scheduler.timesteps
        step_list = timesteps.tolist()
        prev_steps = step_list[1:] + [-1]
        for step_item, prev_item in zip(timesteps, prev_steps):
            predicted = self.get_prediction(latent, step_item, cond)
            variance = self.sample_variance_noise(stream_list, latent)
            latent = self.denoise_posterior_step(latent, predicted, step_item, prev_item, variance)
        return latent

    # One variance draw per row, shaped like the target latent.
    def sample_variance_noise(self, streams: list, target: torch.Tensor):
        row_shape = tuple(target.shape[1:])
        shape = (1,) + row_shape
        row_list = []
        for stream_item in streams:
            row = torch.randn(
                shape, generator=stream_item, device=target.device, dtype=target.dtype
            )
            row_list.append(row)
        variance_noise = torch.cat(row_list)  # (B, C, F, bins)
        return variance_noise

    # One v-prediction step to the x0 estimate.
    @torch.no_grad()
    def denoise_latent_step(
        self, z_t: torch.Tensor, coeff_a: torch.Tensor, coeff_b: torch.Tensor, cond: Conditioning
    ):
        step = self.get_timestep(coeff_a, coeff_b)
        alpha = self.scheduler.alphas_cumprod[step]
        latent = apply_latent_fold(self.spec, z_t, "unfold")
        step_t = torch.tensor(step, device=self.device)
        predicted = self.get_prediction(latent, step_t, cond)
        x0 = self.get_x0_estimate(latent, alpha, predicted)
        folded = apply_latent_fold(self.spec, x0, "fold")  # (B, C*bins, F)
        return folded

    # Sampler denoise from this noise level down to zero.
    @torch.no_grad()
    def denoise_latent_loop(
        self, latent_noisy: torch.Tensor, coeff_a: float, coeff_b: float,
        input_condition: Conditioning, num_steps: int, noise_seeds: list[int],
        cfg_scale: float = 1.0,
    ):
        if coeff_b == 0.0:
            denoised = latent_noisy.to(dtype=torch.float32)
        else:
            coeff_a_t = torch.tensor(coeff_a)
            coeff_b_t = torch.tensor(coeff_b)
            start = self.get_timestep(coeff_a_t, coeff_b_t)
            trajectory = self.make_grid_suffix(self.scheduler, start, num_steps)
            stream_list = []
            for seed_item in noise_seeds:
                stream = torch.Generator(device=self.device)
                seed_value = int(seed_item) + 1
                stream.manual_seed(seed_value)
                stream_list.append(stream)
            cond = replace(input_condition, cfg_scale=cfg_scale)
            latent = apply_latent_fold(self.spec, latent_noisy, "unfold")
            norm = (coeff_a**2 + coeff_b**2) ** 0.5
            latent = latent / norm
            n_traj = len(trajectory)
            for traj_idx, step_item in enumerate(trajectory):
                if traj_idx + 1 < n_traj:
                    prev_step = trajectory[traj_idx + 1]
                else:
                    prev_step = -1
                step_t = torch.tensor(step_item, device=self.device)
                predicted = self.get_prediction(latent, step_t, cond)
                variance = self.sample_variance_noise(stream_list, latent)
                latent = self.denoise_posterior_step(latent, predicted, step_t, prev_step, variance)
            folded = apply_latent_fold(self.spec, latent, "fold")
            denoised = folded.to(dtype=torch.float32)
        return denoised

    # Cached-shard entry point; returns folded latents.
    @torch.no_grad()
    def generate_latent(
        self, text_prompt, diffusion_step: int, seconds_start: float,
        seconds_total: float, seeds: list[int], latent_frames: int,
    ):
        if latent_frames != self.spec.generation_frames:
            raise ValueError(f"tango2 generates {self.spec.generation_frames} frames only")
        if isinstance(text_prompt, str):
            prompts = [text_prompt]
        else:
            prompts = list(text_prompt)
        cond = self.make_conditioning(prompts, seconds_start, seconds_total)
        n_prompts = len(prompts)
        noise = self.sample_noise_vector(n_prompts, seeds)
        latent = self.denoise_diffusion_loop(noise, cond, diffusion_step, seeds)
        folded = apply_latent_fold(self.spec, latent, "fold")  # (B, C*bins, F)
        return folded

    # Generate audio over the fixed window, as both siblings do.
    @torch.no_grad()
    def generate(
        self, text_prompt, diffusion_step: int, seconds_start: float,
        seconds_total: float, seed: int = -1,
    ):
        if isinstance(text_prompt, str):
            prompts = [text_prompt]
        else:
            prompts = list(text_prompt)
        seeds = [seed] * len(prompts)
        latent = self.generate_latent(
            prompts, diffusion_step, seconds_start, seconds_total,
            seeds, self.spec.generation_frames,
        )
        wav = self.decode_latent(latent)
        return wav


# Generation, cache-shaped denoise and batch tensors, with verdict.
@dataclass
class Tango2GeneratorParityResult:
    ours: torch.Tensor | None
    ours_wav: torch.Tensor | None
    denoised: torch.Tensor | None
    batched: torch.Tensor | None
    elapsed: float
    batch_diff: float | None
    passed: bool


# End-to-end generation, cache-shaped denoise, batch invariance.
class Tango2GeneratorParity:
    def __init__(self, generator: Tango2Generator, tokenizer: Tango2Tokenizer,
                 device: str = "cuda"):
        self.generator = generator
        self.tokenizer = tokenizer
        self.device = device
        self.prompts = ["a dog barking"]
        self.steps = 20
        self.seed = 0
        self.ours = None
        self.ours_wav = None
        self.denoised = None
        self.batched = None
        self.elapsed = 0.0
        self.batch_diff = None
        self.result = None

    # Build generator then tokenizer, the old block's order.
    @classmethod
    def make_from_options(cls, device: str = "cuda"):
        generator = Tango2Generator(device=device)
        tokenizer = Tango2Tokenizer(device=device)
        parity = cls(generator, tokenizer, device)
        return parity

    # Generate one latent and check its shapes and finiteness.
    def check_generation(self):
        spec = self.generator.spec
        start = time.perf_counter()
        self.ours = self.generator.generate_latent(
            self.prompts, self.steps, 0.0, 10.24, [self.seed], 256
        )
        self.elapsed = time.perf_counter() - start
        unscaled = self.tokenizer.apply_latent_scale(self.ours, "denormalize")
        self.ours_wav = self.tokenizer.decode(unscaled)

        wanted_latent = (1, spec.latent_dim, 256)
        wanted_wav = (1, spec.audio_channels, 256 * spec.downsampling_ratio)
        latent_shape = tuple(self.ours.shape)
        wav_shape = tuple(self.ours_wav.shape)
        print(f"latent={latent_shape} wav={wav_shape} wall_clock_s={self.elapsed:.2f}")
        abs_latent = self.ours.abs()
        max_latent = abs_latent.max()
        max_latent_value = max_latent.item()
        print(f"max_abs_latent={max_latent_value:.3e}")
        latent_finite = torch.isfinite(self.ours)
        wav_finite = torch.isfinite(self.ours_wav)
        all_finite = bool(latent_finite.all()) and bool(wav_finite.all())
        if latent_shape != wanted_latent:
            print(f"FAIL: SHAPE MISMATCH {latent_shape}")
            passed = False
        elif wav_shape != wanted_wav:
            print(f"FAIL: WAV SHAPE MISMATCH {wav_shape}")
            passed = False
        elif not all_finite:
            print("FAIL: non-finite output")
            passed = False
        else:
            print("PASS: generate_latent returns a finite, cache-shaped latent")
            passed = True
        return passed

    # Denoise a random cache-shaped latent from level 0.5.
    def check_cache_denoise(self):
        spec = self.generator.spec
        cache_latent = torch.randn(1, spec.latent_dim, 256, device=self.device)
        a_t, b_t = get_cosine_noise_coefficients(0.5)
        cond = self.generator.make_conditioning(self.prompts)
        self.denoised = self.generator.denoise_latent_loop(
            cache_latent, a_t, b_t, cond, 5, [self.seed]
        )
        denoised_shape = tuple(self.denoised.shape)
        print(f"cache_shaped_denoise={denoised_shape}")
        wanted_latent = (1, spec.latent_dim, 256)
        finite_mask = torch.isfinite(self.denoised)
        if denoised_shape != wanted_latent:
            print(f"FAIL: CACHE-SHAPE DENOISE MISMATCH {denoised_shape}")
            passed = False
        elif not finite_mask.all():
            print("FAIL: non-finite cache-shaped denoise output")
            passed = False
        else:
            print("PASS: denoise_latent_loop survives a cache-shaped latent")
            passed = True
        return passed

    # Row one of a two-row batch must match solo generation.
    def check_batch_invariance(self):
        batch_prompts = self.prompts + ["a cat meowing"]
        batch_seeds = [self.seed, self.seed + 1]
        self.batched = self.generator.generate_latent(
            batch_prompts, self.steps, 0.0, 10.24, batch_seeds, 256
        )
        first_row = self.batched[:1]
        gap = self.ours - first_row
        abs_gap = gap.abs()
        max_gap = abs_gap.max()
        self.batch_diff = max_gap.item()
        print(f"batch_invariance_max_abs_diff={self.batch_diff:.3e}")
        if self.batch_diff != 0.0:
            print(f"FAIL: BATCH VARIANCE detected max_abs_diff={self.batch_diff}")
            passed = False
        else:
            print("PASS: row 1 of a 2-row batch matches the solo generation")
            passed = True
        return passed

    # Run the three checks in order, stopping at first failure.
    def run(self):
        passed = self.check_generation()
        if passed:
            passed = self.check_cache_denoise()
        if passed:
            passed = self.check_batch_invariance()
        self.result = Tango2GeneratorParityResult(
            ours=self.ours,
            ours_wav=self.ours_wav,
            denoised=self.denoised,
            batched=self.batched,
            elapsed=self.elapsed,
            batch_diff=self.batch_diff,
            passed=passed,
        )
        return self.result


@click.command()
@click.option("--data_root", type=Path, required=True)
def main(data_root):
    apply_data_root(data_root)
    parity = Tango2GeneratorParity.make_from_options()
    result = parity.run()
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
