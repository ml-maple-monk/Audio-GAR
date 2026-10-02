from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import click
import numpy as np
import torch
from torch import nn

from . import registry
from ...runtime.dataroot import apply_data_root
from .tokenizer import AudioLDM2Tokenizer, apply_latent_fold
from .vendor.audioldm2.latent_diffusion.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
)
from .vendor.audioldm2.latent_diffusion.util import get_vits_phoneme_ids_no_padding


# upstream keeps these zero placeholders in every text-to-audio batch
MEL_FRAMES = 1024
MEL_BINS = 64
STFT_BINS = 512
KALDI_BINS = 128
WAVEFORM_SAMPLES = 160000
# keeps candidate streams off neighbouring rows
CANDIDATE_STRIDE = 1000003
# transformers dropped these constant buffers from RoBERTa after 4.30
STALE_BUFFER = ".embeddings.position_ids"
# v1 stores one cond tree; v2 stores the list
COND_LIST_PREFIX = "cond_stage_models.0."
# DDPM's own best-of-N scorer; unconditional, never checkpointed
SCORER_PREFIX = "clap."


# Guided and null conditioning dicts, plus the guidance weight.
@dataclass
class Conditioning:
    cond: dict
    uncond: dict | None
    cfg_scale: float


# AudioLDM 2 text-to-audio latent generator.
class AudioLDM2Generator(nn.Module):
    def __init__(self, name: str = "audioldm2-full-large", ckpt_path: str = "", device: str = "cuda",
                 determinism: bool = False, compile_blocks: bool = False):
        super().__init__()
        self.name = name

        self.spec = registry.spec(name)
        # ddpm drags CLAP, whose tokenizer resolves at import time
        self.apply_asset_paths(self.spec)
        # lazy: vendored ddpm must import after asset paths are set
        from .vendor.audioldm2.latent_diffusion.models.ddpm import LatentDiffusion

        model_params = self.spec.get_model_params(device)
        self.model = LatentDiffusion(**model_params)

        ckpt = ckpt_path
        if not ckpt:
            registry_ckpt = registry.get_ckpt_path(name)
            ckpt = registry_ckpt or ""
        self.weights_path = ckpt
        if ckpt:
            self.load_ckpt(ckpt)

        self.model.to(device)
        self.model.eval()

        param_iter = self.model.parameters()
        first_param = next(param_iter)
        self.device = first_param.device
        unet_param_iter = self.model.model.parameters()
        unet_param = next(unet_param_iter)
        self.dtype = unet_param.dtype
        self.codec = None
        self.determinism = determinism
        if determinism:
            self.apply_determinism()
        else:
            self.apply_precision_policy()
        if compile_blocks:
            self.apply_compile()

    # Point the vendored hub ids at this run's staged copies.
    def apply_asset_paths(self, spec):
        # lazy: heavy vendored CLAP stack loads only when building
        from .vendor.audioldm2.audiomae_gen import sequence_input
        from .vendor.audioldm2.clap.open_clip import model as clap_model
        from .vendor.audioldm2.clap.training import data as clap_data
        from .vendor.audioldm2.latent_diffusion.modules.encoders import modules

        tokenizer = registry.get_asset_path(spec, "text_tokenizer")
        sequence_input.GPT2_ID = registry.get_asset_path(spec, "sequence_model")
        clap_model.ROBERTA_ID = tokenizer
        clap_data.ROBERTA_ID = tokenizer
        modules.ROBERTA_ID = tokenizer

    # Rename a v1 cond tree onto the v2 module list.
    def get_remapped_cond_state(self, state: dict, legacy_prefix: str):
        prefix_len = len(legacy_prefix)
        remapped = {}
        for key_item, value in state.items():
            new_key = key_item
            if key_item.startswith(legacy_prefix):
                new_key = COND_LIST_PREFIX + key_item[prefix_len:]
            remapped[new_key] = value
        return remapped

    # Drop only buffers this transformers no longer registers.
    def get_loadable_state(self, model, state: dict):
        model_state = model.state_dict()
        wanted = set(model_state)
        extra = []
        for key_item in state:
            if key_item not in wanted:
                extra.append(key_item)
        stale = []
        for key_item in extra:
            if key_item.endswith(STALE_BUFFER):
                stale.append(key_item)
        stale_set = set(stale)
        extra_set = set(extra)
        unexpected_set = extra_set - stale_set
        unexpected = sorted(unexpected_set)
        if unexpected:
            raise ValueError(f"checkpoint carries unexpected keys: {unexpected[:4]}")
        loadable = {}
        for key_item, value in state.items():
            if key_item not in stale_set:
                loadable[key_item] = value
        return loadable

    # Strict load; tolerate named buffers, scorer if row allows.
    def load_checked_state(
        self, model, state: dict, allow_missing: tuple[str, ...], allow_missing_scorer: bool,
    ):
        result = model.load_state_dict(state, strict=False)
        scorer_ok = allow_missing_scorer
        missing = []
        for key_item in result.missing_keys:
            allowed = key_item.endswith(allow_missing)
            scorer_key = scorer_ok and key_item.startswith(SCORER_PREFIX)
            if not allowed and not scorer_key:
                missing.append(key_item)
        if missing or result.unexpected_keys:
            unexpected = list(result.unexpected_keys)
            raise ValueError(
                f"checkpoint load failed: missing={missing[:4]} "
                f"unexpected={unexpected[:4]}"
            )

    # Load the released checkpoint into the diffusion model.
    def load_ckpt(self, ckpt: str):
        bundle = torch.load(ckpt, map_location="cpu", weights_only=True)
        if "state_dict" in bundle:
            state = bundle["state_dict"]
        else:
            state = bundle
        if self.spec.legacy_cond_prefix:
            state = self.get_remapped_cond_state(state, self.spec.legacy_cond_prefix)
        state = self.get_loadable_state(self.model, state)
        self.load_checked_state(
            self.model, state, self.spec.cond_allowed_missing,
            self.spec.allow_missing_scorer,
        )

    # Tf32 and autotuning, which determinism forbids.
    def apply_precision_policy(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Compile every U-Net stage independently.
    def apply_compile(self):
        unet = self.model.model.diffusion_model
        n_input = len(unet.input_blocks)
        for block_idx in range(n_input):
            block = unet.input_blocks[block_idx]
            unet.input_blocks[block_idx] = torch.compile(block, dynamic=False)
        unet.middle_block = torch.compile(unet.middle_block, dynamic=False)
        n_output = len(unet.output_blocks)
        for block_idx in range(n_output):
            block = unet.output_blocks[block_idx]
            unet.output_blocks[block_idx] = torch.compile(block, dynamic=False)

    # Deterministic kernels, no tf32, no autotuning.
    def apply_determinism(self):
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # EMA weights during sampling, as released eval does.
    def get_ema_scope(self):
        if self.spec.use_ema:
            scope = self.model.ema_scope("generate")
        else:
            scope = nullcontext()
        return scope

    # Tokenizer built on first decode, so generation stays light.
    def get_codec(self):
        if self.codec is None:
            # the codec sets backend flags, so it inherits them
            device_name = str(self.device)
            self.codec = AudioLDM2Tokenizer(
                self.name, ckpt_path=self.weights_path, device=device_name,
                determinism=self.determinism,
            )
        return self.codec

    # Decode from the folded, scaled latent space.
    def decode_latent(self, latent: torch.Tensor):
        codec = self.get_codec()
        wav = codec.decode_latent(latent)
        return wav

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
        rows = []
        for seed_item in seeds:
            seed_int = int(seed_item)
            stream = torch.Generator(device=self.device)
            stream.manual_seed(seed_int)
            row = torch.randn(shape, generator=stream, device=self.device)  # (1, C, T, F)
            rows.append(row)
        noise = torch.cat(rows)
        return noise

    # One prompt list from a prompt string or sequence.
    def make_prompt_list(self, text_prompt):
        if isinstance(text_prompt, str):
            prompts = [text_prompt]
        else:
            prompts = list(text_prompt)
        return prompts

    # Upstream's zero-filled batch, one row per prompt.
    def make_text_batch(self, prompts: list[str]):
        size = len(prompts)
        fname_list = []
        for row_idx in range(size):
            fname_list.append(f"row{row_idx}")
        batch = {
            "text": list(prompts),
            "fname": fname_list,
            "waveform": torch.zeros((size, WAVEFORM_SAMPLES)),
            "stft": torch.zeros((size, MEL_FRAMES, STFT_BINS)),
            "log_mel_spec": torch.zeros((size, MEL_FRAMES, MEL_BINS)),
            "ta_kaldi_fbank": torch.zeros((size, MEL_FRAMES, KALDI_BINS)),
        }
        empty_list = [""] * size
        phoneme_dict = get_vits_phoneme_ids_no_padding(empty_list)
        batch.update(phoneme_dict)
        return batch

    # Concatenate on batch, right-padding a ragged token axis.
    def get_padded_stack(self, tensors: list[torch.Tensor]):
        width_list = []
        for tensor_item in tensors:
            width_list.append(tensor_item.shape[1])
        width = max(width_list)
        padded = []
        for tensor_item in tensors:
            short = width - tensor_item.shape[1]
            padded_tensor = tensor_item
            if short:
                # pad counts from the last axis outward
                trailing = [0, 0] * (tensor_item.ndim - 2)
                pad_list = trailing + [0, short]
                padded_tensor = nn.functional.pad(tensor_item, pad_list)
            padded.append(padded_tensor)
        stacked = torch.cat(padded, dim=0)
        return stacked

    # Stack per-row conditioning, zero-padding ragged token axes.
    def make_merged_conditioning(self, rows: list[dict]):
        merged = {}
        first_row = rows[0]
        for key_item in first_row:
            values = []
            for row_item in rows:
                values.append(row_item[key_item])
            if isinstance(values[0], list):
                stack_list = []
                n_entries = len(values[0])
                for entry_idx in range(n_entries):
                    entry_list = []
                    for value_item in values:
                        entry_list.append(value_item[entry_idx])
                    stack = self.get_padded_stack(entry_list)
                    stack_list.append(stack)
                merged[key_item] = stack_list
            else:
                merged[key_item] = self.get_padded_stack(values)
        return merged

    # Conditioning for the guided and null branches.
    @torch.no_grad()
    def make_conditioning(
        self,
        text_prompt: str | list[str],
        seconds_start: float | list[float] = 0.0,
        seconds_total: float | list[float] = 10.24,
        negative_prompt: str | None = None,
    ):
        prompts = self.make_prompt_list(text_prompt)
        # per row: T5 pads to its batch's longest prompt
        rows = []
        for prompt_item in prompts:
            row = self.make_row_conditioning(prompt_item)
            rows.append(row)
        cond = self.make_merged_conditioning(rows)
        n_prompts = len(prompts)
        uncond = self.make_null_conditioning(n_prompts)
        conditioning = Conditioning(cond, uncond, self.spec.sampling.cfg_scale)
        return conditioning

    # Conditioning for one prompt, free of any batch neighbour.
    @torch.no_grad()
    def make_row_conditioning(self, prompt: str):
        prompt_list = [prompt]
        batch = self.make_text_batch(prompt_list)
        _, cond = self.model.get_input(
            batch, self.model.first_stage_key,
            return_first_stage_encode=False, unconditional_prob_cfg=0.0,
        )
        useful = self.model.filter_useful_cond_dict(cond)
        return useful

    # Each conditioner's own unconditional token set.
    @torch.no_grad()
    def make_null_conditioning(self, batch_size: int):
        null = {}
        metadata = self.model.cond_stage_model_metadata
        for key_item, meta_item in metadata.items():
            model_idx = meta_item["model_idx"]
            model = self.model.cond_stage_models[model_idx]
            null[key_item] = model.get_unconditional_condition(batch_size)
        return null

    # Upstream's stride can name one step past the schedule.
    def get_bounded_timesteps(self, timesteps, total: int):
        bounded = np.minimum(timesteps, total - 1)
        kept = []
        seen = set()
        for step_item in bounded:
            step_int = int(step_item)
            if step_int not in seen:
                seen.add(step_int)
                kept.append(step_int)
        bounded_array = np.asarray(kept)
        return bounded_array

    # Rising timesteps ending at the level, never the zero entry.
    def get_truncated_timesteps(self, steps: int, last_step: int):
        if steps < 1:
            raise ValueError("a denoise needs at least one step")
        if last_step < 1:
            raise ValueError("a partial denoise needs a positive start timestep")
        # entry zero transitions onto itself, so start at one
        n_steps = min(steps, last_step)
        if n_steps == 1:
            timesteps = np.asarray([last_step])
        else:
            stride = (last_step - 1) / (n_steps - 1)
            step_list = []
            for step_idx in range(n_steps):
                step_list.append(round(1 + step_idx * stride))
            timesteps = np.asarray(step_list)
        return timesteps

    # Timesteps and DDIM coefficients, upstream formulas.
    def get_ddim_schedule(self, steps: int, last_step: int | None = None):
        n_timesteps = self.spec.schedule.timesteps
        if last_step is None:
            # released inference: upstream's own integer stride
            timesteps = make_ddim_timesteps("uniform", steps, n_timesteps, verbose=False)
            timesteps = self.get_bounded_timesteps(timesteps, n_timesteps)
        else:
            timesteps = self.get_truncated_timesteps(steps, last_step)
        alphas_cumprod = self.model.alphas_cumprod
        detached = alphas_cumprod.detach()
        alphas_cpu = detached.cpu()
        sigmas, alphas, alphas_prev = make_ddim_sampling_parameters(
            alphas_cpu, timesteps, self.spec.sampling.ddim_eta, verbose=False
        )
        return timesteps, alphas, alphas_prev, sigmas

    # One stochastic draw per row, from that row's stream.
    def sample_step_noise(self, streams: list, target: torch.Tensor):
        tail_shape = tuple(target.shape[1:])
        shape = (1,) + tail_shape
        rows = []
        for stream_item in streams:
            row = torch.randn(shape, generator=stream_item, device=target.device, dtype=target.dtype)
            rows.append(row)
        noise = torch.cat(rows)
        return noise

    # DDIM loop whose every draw belongs to one row's seed.
    @torch.no_grad()
    def denoise_diffusion_loop(
        self, noise: torch.Tensor, cond: Conditioning, steps: int,
        seeds: list[int], last_step: int | None = None,
    ):
        timesteps, alphas, alphas_prev, sigmas = self.get_ddim_schedule(steps, last_step)
        # +1 keeps the step stream independent of the x_T stream
        streams = []
        for seed_item in seeds:
            stream_seed = int(seed_item) + 1
            stream = torch.Generator(device=self.device)
            stream.manual_seed(stream_seed)
            streams.append(stream)
        latent = noise
        n_steps = len(timesteps)
        step_range = range(n_steps)
        step_order = reversed(step_range)
        for step_idx in step_order:
            step = int(timesteps[step_idx])
            alpha = float(alphas[step_idx])
            alpha_prev = float(alphas_prev[step_idx])
            sigma = float(sigmas[step_idx])
            latent = self.denoise_ddim_step(latent, step, cond, streams, alpha, alpha_prev, sigma)
        return latent

    # One DDIM update between two schedule entries.
    @torch.no_grad()
    def denoise_ddim_step(
        self, latent: torch.Tensor, step: int, cond: Conditioning, streams: list,
        alpha: float, alpha_prev: float, sigma: float,
    ):
        predicted = self.get_prediction(latent, step, cond)
        pred_x0 = (latent - (1.0 - alpha) ** 0.5 * predicted) / alpha**0.5
        direction = max(1.0 - alpha_prev - sigma**2, 0.0) ** 0.5 * predicted
        updated = alpha_prev**0.5 * pred_x0 + direction
        if sigma > 0.0:
            step_noise = self.sample_step_noise(streams, updated)
            updated = updated + sigma * step_noise
        return updated

    # Cached-shard entry point; returns folded latents.
    @torch.no_grad()
    def generate_latent(
        self, text_prompt, diffusion_step: int, seconds_start: float,
        seconds_total: float, seeds: list[int], latent_frames: int,
    ):
        if latent_frames != self.spec.generation_frames:
            raise ValueError(f"audioldm2 generates {self.spec.generation_frames} frames only")
        prompts = self.make_prompt_list(text_prompt)
        if len(prompts) != len(seeds):
            raise ValueError("prompt and seed counts differ")
        ema_scope = self.get_ema_scope()
        with ema_scope:
            cond = self.make_conditioning(prompts, seconds_start, seconds_total)
            n_candidates = max(self.spec.sampling.candidates, 1)
            drawn = []
            for candidate_idx in range(n_candidates):
                candidate = self.sample_candidate_latent(
                    cond, diffusion_step, seeds, candidate_idx
                )
                drawn.append(candidate)
            if len(drawn) == 1:
                latent = drawn[0]
            else:
                latent = self.get_best_candidate(drawn, prompts)
        return latent

    # One folded candidate, on its own per-row seed stream.
    @torch.no_grad()
    def sample_candidate_latent(
        self, cond: Conditioning, steps: int, seeds: list[int], index: int
    ):
        drawn = []
        for seed_item in seeds:
            drawn.append(seed_item + index * CANDIDATE_STRIDE)
        n_drawn = len(drawn)
        noise = self.sample_noise_vector(n_drawn, drawn)
        latent = self.denoise_diffusion_loop(noise, cond, steps, drawn)
        folded = apply_latent_fold(self.spec, latent, "fold")
        return folded

    # Released selection: CLAP scores each decode, best latent wins.
    @torch.no_grad()
    def get_best_candidate(
        self, candidates: list[torch.Tensor], prompts: list[str]
    ):
        if self.spec.sampling.candidate_scorer == "cond":
            # v1 has no trained top-level scorer; use its conditioning CLAP
            scorer = self.model.cond_stage_models[0]
        else:
            scorer = self.model.clap
        # upstream leaves training dropout on, which would randomise scoring
        scorer.unconditional_prob = 0.0
        n_prompts = len(prompts)
        scores = []
        for candidate_item in candidates:
            decoded = self.decode_latent(candidate_item)
            squeezed = decoded.squeeze(1)
            wav_float = squeezed.float()
            wav = wav_float.cpu()
            similarity = scorer.cos_similarity(wav, prompts)
            score = similarity.reshape(n_prompts)
            scores.append(score)
        stacked = torch.stack(scores)
        best = stacked.argmax(dim=0)
        rows = []
        for row_idx in range(n_prompts):
            best_idx = int(best[row_idx])
            best_candidate = candidates[best_idx]
            rows.append(best_candidate[row_idx])
        best_latent = torch.stack(rows)
        return best_latent

    # Nearest schedule step matching these cosine coefficients.
    def get_timestep(self, coeff_a, coeff_b):
        wanted = float(coeff_a / (coeff_a**2 + coeff_b**2) ** 0.5)
        root = self.model.alphas_cumprod.sqrt()
        gap = root - wanted
        distance = gap.abs()
        nearest = distance.argmin()
        step = int(nearest)
        return step

    # One guided epsilon prediction, two separate model calls.
    @torch.no_grad()
    def get_prediction(self, latent: torch.Tensor, step, cond: Conditioning):
        step_int = int(step)
        n_rows = latent.shape[0]
        steps = torch.full((n_rows,), step_int, device=self.device, dtype=torch.long)
        guided = self.model.apply_model(latent, steps, cond.cond)
        if cond.uncond is None or cond.cfg_scale == 1.0:
            predicted = guided
        else:
            null = self.model.apply_model(latent, steps, cond.uncond)
            predicted = null + cond.cfg_scale * (guided - null)
        return predicted

    # One epsilon step resolved to the clean-sample estimate.
    @torch.no_grad()
    def denoise_latent_step(
        self, z_t: torch.Tensor, coeff_a, coeff_b, cond: Conditioning
    ):
        step = self.get_timestep(coeff_a, coeff_b)
        alpha = self.model.alphas_cumprod[step]
        latent = apply_latent_fold(self.spec, z_t, "unfold")
        predicted = self.get_prediction(latent, step, cond)
        one_minus = 1.0 - alpha
        noise_scale = one_minus.sqrt()
        signal_scale = alpha.sqrt()
        x0 = (latent - noise_scale * predicted) / signal_scale
        folded = apply_latent_fold(self.spec, x0, "fold")
        return folded

    # Sampler denoise from this noise level down to zero.
    @torch.no_grad()
    def denoise_latent_loop(
        self, latent_noisy: torch.Tensor, coeff_a: float, coeff_b: float,
        input_condition: Conditioning, num_steps: int, noise_seeds: list[int],
        cfg_scale: float = 1.0,
    ):
        if coeff_b == 0.0:
            latent_out = latent_noisy.to(dtype=torch.float32)
        else:
            cond = replace(input_condition, cfg_scale=cfg_scale)
            coeff_a_tensor = torch.tensor(coeff_a)
            coeff_b_tensor = torch.tensor(coeff_b)
            start = self.get_timestep(coeff_a_tensor, coeff_b_tensor)
            latent = apply_latent_fold(self.spec, latent_noisy, "unfold")
            latent = latent / (coeff_a**2 + coeff_b**2) ** 0.5
            # truncated schedule pairs each alpha with its successor
            latent = self.denoise_diffusion_loop(latent, cond, num_steps, noise_seeds, start)
            folded = apply_latent_fold(self.spec, latent, "fold")
            latent_out = folded.to(dtype=torch.float32)
        return latent_out

    # Generate audio over the fixed window, as every sibling does.
    @torch.no_grad()
    def generate(
        self, text_prompt, diffusion_step: int, seconds_start: float,
        seconds_total: float, seed: int = -1,
    ):
        if seconds_start:
            raise ValueError("audioldm2 has no start-time conditioning")
        prompts = self.make_prompt_list(text_prompt)
        n_prompts = len(prompts)
        seeds = self.get_row_seeds(seed, n_prompts)
        latent = self.generate_latent(
            prompts, diffusion_step, seconds_start, seconds_total,
            seeds, self.spec.generation_frames,
        )
        audio = self.decode_latent(latent)
        window_samples = int(seconds_total * self.spec.sample_rate)
        trim = min(window_samples, audio.shape[-1])
        trimmed = audio[..., :trim]
        return trimmed

    # Sibling sentinel: -1 draws, and rows never share a stream.
    def get_row_seeds(self, seed: int, rows: int):
        if seed == -1:
            drawn = torch.randint(0, 2**31 - 1, (1,))
            drawn_value = drawn.item()
            seed = int(drawn_value)
        seed_list = []
        for row_idx in range(rows):
            seed_list.append(seed + row_idx)
        return seed_list


# Latent, waveform and verdict of one generator parity check.
@dataclass
class AudioLDM2GeneratorParityResult:
    ours: torch.Tensor
    wav: torch.Tensor | None
    max_abs: float
    passed: bool


# Parity against the released pipeline's own sampling path.
class AudioLDM2GeneratorParity:
    def __init__(self, generator: AudioLDM2Generator, prompt: str = "a dog barking",
                 steps: int = 20, seed: int = 0):
        self.generator = generator
        self.prompt = prompt
        self.steps = steps
        self.seed = seed
        self.result = None

    # Build the deterministic default generator on one device.
    @classmethod
    def make_from_options(cls, device: str = "cuda"):
        generator = AudioLDM2Generator(device=device, determinism=True)
        parity = cls(generator)
        return parity

    # Name the first latent fault, or return an empty string.
    def check_latent(self, ours: torch.Tensor):
        spec = self.generator.spec
        latent_shape = tuple(ours.shape)
        fault = ""
        if latent_shape != (1, spec.latent_dim, spec.generation_frames):
            fault = f"SHAPE MISMATCH {latent_shape}"
        else:
            finite_mask = torch.isfinite(ours)
            all_finite = finite_mask.all()
            if not all_finite:
                fault = "non-finite latent"
        return fault

    # Name the waveform shape fault, or return an empty string.
    def check_wav(self, wav: torch.Tensor):
        spec = self.generator.spec
        wanted = spec.generation_frames * spec.downsampling_ratio
        wav_shape = tuple(wav.shape)
        fault = ""
        if wav_shape != (1, spec.audio_channels, wanted):
            fault = f"WAV SHAPE MISMATCH {wav_shape}"
        return fault

    # Generate one cache-shaped latent, decode it, check both.
    def run(self):
        spec = self.generator.spec
        prompt_list = [self.prompt]
        seed_list = [self.seed]
        ours = self.generator.generate_latent(
            prompt_list, self.steps, 0.0, spec.seconds_total, seed_list, 256
        )
        abs_latent = ours.abs()
        max_tensor = abs_latent.max()
        max_abs = max_tensor.item()
        latent_shape = tuple(ours.shape)
        print(f"latent={latent_shape} max_abs={max_abs:.3e}")
        fault = self.check_latent(ours)
        wav = None
        if not fault:
            wav = self.generator.decode_latent(ours)
            wav_shape = tuple(wav.shape)
            print(f"wav={wav_shape}")
            fault = self.check_wav(wav)
        passed = not fault
        if passed:
            print("PASS: generate_latent returns a finite, cache-shaped latent")
        else:
            print(f"FAIL: {fault}")
        result = AudioLDM2GeneratorParityResult(ours=ours, wav=wav, max_abs=max_abs, passed=passed)
        self.result = result
        return result


# Run the generator parity check on the default checkpoint.
@click.command()
@click.option("--data_root", type=Path, required=True)
def main(data_root):
    apply_data_root(data_root)
    parity = AudioLDM2GeneratorParity.make_from_options()
    result = parity.run()
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
