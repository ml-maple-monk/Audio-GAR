from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import torch
from k_diffusion.external import VDenoiser
from k_diffusion.sampling import (
    BrownianTreeNoiseSampler,
    get_sigmas_polyexponential,
    sample_dpmpp_3m_sde,
)
from torch import nn

from ...runtime.dataroot import apply_data_root
from ..generator import get_schedule_sigmas
from . import registry
from .vendor.generator_arch import (
    AudioAutoencoder,
    AutoencoderPretransform,
    ConditionedDiffusionModelWrapper,
    DiTWrapper,
    OobleckDecoder,
    OobleckEncoder,
    VAEBottleneck,
    flash_attn_func,
    generate_diffusion_cond,
    load_ckpt_state_dict,
)


VAE_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
COMPILE_MODES = ("off", "blocks", "model", "cudagraph")


@dataclass
class Conditioning:
    text_timing: torch.Tensor
    text_timing_mask: torch.Tensor
    timing: torch.Tensor
    negative_text_timing: torch.Tensor | None = None
    negative_text_timing_mask: torch.Tensor | None = None
    negative_timing: torch.Tensor | None = None


@dataclass
class DiffusionCondInputs:
    cross_attn_cond: torch.Tensor
    cross_attn_mask: torch.Tensor
    global_cond: torch.Tensor | None
    negative_cross_attn_cond: torch.Tensor | None
    negative_cross_attn_mask: torch.Tensor | None
    negative_global_cond: torch.Tensor | None
    cfg_scale: float
    rescale_cfg: bool
    batch_cfg: bool


@dataclass
class ConditioningInput:
    video_tensors: torch.Tensor
    text_prompt: str
    audio_prompt: torch.Tensor

    # Upstream conditioning metadata for one clip.
    def get_metadata(self):
        metadata = {
            "video_prompt": {"video_tensors": self.video_tensors, "video_sync_frames": None},
            "text_prompt": self.text_prompt,
            "audio_prompt": self.audio_prompt,
        }
        return metadata


# L2 wrapper for base AudioX.
class AudioXGenerator(nn.Module):
    def __init__(self, name: str = "audiox", ckpt_path: str = "", device: str = "cuda",
                 determinism: bool = False, vae_dtype: str = "float32",
                 compile_mode: str = "off"):
        super().__init__()
        self.name = name

        self.spec = registry.spec(name)
        self.gen_model = self.make_gen_model()

        ckpt = ckpt_path or registry.get_ckpt_path(name)
        if ckpt:
            state_dict = load_ckpt_state_dict(ckpt)
            self.gen_model.load_state_dict(state_dict, strict=True)

        self.gen_model.to(device)
        self.gen_model.eval()

        param_iter = self.gen_model.parameters()
        first_param = next(param_iter)
        self.device = first_param.device
        model_param_iter = self.gen_model.model.parameters()
        first_model_param = next(model_param_iter)
        self.dtype = first_model_param.dtype
        self.check_flash_kernels(self.device)
        if determinism:
            self.apply_determinism()
        else:
            self.apply_precision_policy()
        self.apply_vae_dtype(vae_dtype)
        self.apply_compile(compile_mode)

    # Oobleck VAE autoencoder, wrapped as the diffusion pretransform.
    def make_pretransform(self):
        encoder = OobleckEncoder(
            in_channels=self.spec.audio_channels,
            channels=self.spec.vae.channels,
            latent_dim=2 * self.spec.latent_dim,
            c_mults=self.spec.vae.c_mults,
            strides=self.spec.vae.strides,
            use_snake=self.spec.vae.use_snake,
        )
        decoder = OobleckDecoder(
            out_channels=self.spec.audio_channels,
            channels=self.spec.vae.channels,
            latent_dim=self.spec.latent_dim,
            c_mults=self.spec.vae.c_mults,
            strides=self.spec.vae.strides,
            use_snake=self.spec.vae.use_snake,
            final_tanh=self.spec.vae.final_tanh,
        )
        bottleneck = VAEBottleneck()
        autoencoder = AudioAutoencoder(
            encoder,
            decoder,
            latent_dim=self.spec.latent_dim,
            downsampling_ratio=self.spec.downsampling_ratio,
            sample_rate=self.spec.sample_rate,
            io_channels=self.spec.audio_channels,
            bottleneck=bottleneck,
        )

        pretransform = AutoencoderPretransform(
            autoencoder,
            scale=self.spec.vae.scale,
            iterate_batch=self.spec.vae.iterate_batch,
        )
        return pretransform

    # Diffusion transformer backbone.
    def make_dit(self):
        dit = DiTWrapper(
            io_channels=self.spec.latent_dim,
            embed_dim=self.spec.dit.embed_dim,
            depth=self.spec.dit.depth,
            num_heads=self.spec.dit.num_heads,
            cond_token_dim=self.spec.dit.cond_token_dim,
            global_cond_dim=self.spec.dit.global_cond_dim,
            project_cond_tokens=self.spec.dit.project_cond_tokens,
            transformer_type=self.spec.dit.transformer_type,
        )
        return dit

    # Assemble the conditioned diffusion wrapper from its three parts.
    def make_gen_model(self):
        pretransform = self.make_pretransform()
        dit = self.make_dit()
        conditioner = self.spec.make_conditioner()
        gate_type_config = self.spec.gate.get_type_config()

        gen_model = ConditionedDiffusionModelWrapper(
            model=dit,
            conditioner=conditioner,
            pretransform=pretransform,
            io_channels=self.spec.latent_dim,
            sample_rate=self.spec.sample_rate,
            min_input_length=self.spec.downsampling_ratio,
            diffusion_objective=self.spec.dit.diffusion_objective,
            gate=self.spec.gate.enabled,
            gate_type=self.spec.gate.gate_type,
            gate_type_config=gate_type_config,
            cross_attn_cond_ids=self.spec.conditioning.cross_attn_cond_ids,
            global_cond_ids=self.spec.conditioning.global_cond_ids,
        )
        if self.spec.gate.enabled and self.spec.gate.gate_type == "MAF":
            # the MAF port pulls torchaudio, absent on some images
            from .vendor.maf import MAFBlock

            gen_model.maf_block = MAFBlock(
                dim=self.spec.conditioning.cond_dim,
                num_experts_per_modality=self.spec.gate.num_experts_per_modality,
                num_heads=self.spec.gate.num_heads,
                num_fusion_layers=self.spec.gate.num_fusion_layers,
            )
        return gen_model

    # Block list the sampler re-enters every step.
    def get_transformer_blocks(self, model):
        blocks = None
        for module_item in model.modules():
            layers = getattr(module_item, "layers", None)
            if isinstance(layers, nn.ModuleList) and len(layers) > 1:
                blocks = layers
                break
        return blocks

    # Fail loud when the vendored flash attention path is absent.
    def check_flash_kernels(self, device):
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and flash_attn_func is None:
            raise Exception("flash_attn is not importable; the DiT requires it on cuda")

    # Tf32 and autotuning, which determinism forbids.
    def apply_precision_policy(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Recast the codec after its checkpoint has loaded.
    def apply_vae_dtype(self, vae_dtype: str):
        target = VAE_DTYPES[vae_dtype]
        if target is not torch.float32:
            pretransform_cast = self.gen_model.pretransform.to(dtype=target)
            self.gen_model.pretransform = pretransform_cast

    # Whole DiT as one graph; blocks is the legacy path.
    def apply_compile(self, compile_mode: str):
        if compile_mode not in COMPILE_MODES:
            raise ValueError(f"unknown compile_mode {compile_mode!r}")
        if compile_mode == "blocks":
            blocks = self.get_transformer_blocks(self.gen_model.model)
            n_blocks = len(blocks)
            for block_idx in range(n_blocks):
                block = blocks[block_idx]
                compiled_block = torch.compile(block, dynamic=False)
                blocks[block_idx] = compiled_block
        elif compile_mode != "off":
            # cudagraph replays the DiT; the sampler math stays eager
            if compile_mode == "cudagraph":
                mode = "reduce-overhead"
            else:
                mode = "default"
            compiled_model = torch.compile(self.gen_model.model, dynamic=False, mode=mode)
            self.gen_model.model = compiled_model

    # Pin every backend switch that varies run to run.
    def apply_determinism(self):
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        # bf16 twin defaults on and every forward here is bf16
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        if torch.version.hip:
            # hipBLASLt picks a new algorithm per process, rocBLAS does not
            torch.backends.cuda.preferred_blas_library("cublas")

    # Text encodes; absent video and audio follow the released path.
    @torch.no_grad()
    def make_conditioning_sources(self, prompts: list, seconds_total: float):
        parts = self.gen_model.conditioner.conditioners
        count = len(prompts)
        video_feat = parts["video_prompt"].empty_visual_feat.detach()
        video_expanded = video_feat.expand(count, -1, -1)
        video_feat = video_expanded.contiguous()
        if self.spec.conditioning.audio_type == "mel_spec":
            samples = int(self.spec.sample_rate * seconds_total)
            zero_audio = []
            for prompt_item in prompts:
                zero_wav = torch.zeros(
                    1,
                    self.spec.audio_channels,
                    samples,
                    device=self.device,
                    dtype=torch.float32,
                )  # (1, C, T)
                zero_audio.append(zero_wav)
            audio_feat, audio_mask = parts["audio_prompt"](zero_audio, self.device)
        else:
            audio_feat = parts["audio_prompt"].empty_audio_feat.detach()
            audio_expanded = audio_feat.expand(count, -1, -1)
            audio_feat = audio_expanded.contiguous()
            audio_mask = torch.ones(count, audio_feat.shape[2], device=self.device)
        text_feat, text_mask = parts["text_prompt"](prompts, self.device)
        video_mask = torch.ones(count, 1, device=self.device)
        sources = {
            "video_prompt": (video_feat, video_mask),
            "text_prompt": (text_feat, text_mask),
            "audio_prompt": [audio_feat, audio_mask],
        }
        return sources

    # Wrapper conditioning inputs for one prompt batch.
    def make_conditioning_inputs(self, prompt: str | list[str], is_negative: bool,
                                 seconds_total: float):
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt
        sources = self.make_conditioning_sources(prompts, seconds_total)
        inputs = self.gen_model.get_conditioning_inputs(sources, negative=is_negative)
        return inputs

    # Build text-to-audio conditioning once.
    @torch.no_grad()
    def make_conditioning(
        self,
        text_prompt: str | list[str],
        seconds_start: float,
        seconds_total: float,
        negative_prompt: str | None = None,
    ):
        positive = self.make_conditioning_inputs(text_prompt, is_negative=False,
                                                  seconds_total=seconds_total)
        cond = Conditioning(
            text_timing=positive["cross_attn_cond"],
            text_timing_mask=positive["cross_attn_mask"],
            timing=positive["global_cond"],
        )
        if negative_prompt is not None:
            negative = self.make_conditioning_inputs(negative_prompt, is_negative=True,
                                                      seconds_total=seconds_total)
            cond.negative_text_timing = negative["negative_cross_attn_cond"]
            cond.negative_text_timing_mask = negative["negative_cross_attn_mask"]
            cond.negative_timing = negative["negative_global_cond"]
        return cond

    # Sample initial latent noise; deterministic when seeded.
    def sample_noise_vector(
        self, batch_size: int = 1, seed: int = -1, latent_frames: int | None = None,
    ):
        downsampling_ratio = self.gen_model.pretransform.downsampling_ratio
        latent_length = latent_frames or self.spec.sample_size // downsampling_ratio

        if seed == -1:
            seed_tensor = torch.randint(0, 2**32 - 1, (1,))
            seed_value = seed_tensor.item()
            seed = int(seed_value)
        torch.manual_seed(seed)

        noise = torch.randn(
            [batch_size, self.gen_model.io_channels, latent_length],
            device=self.device,
        )  # (B, D, F)
        noise = noise.to(dtype=self.dtype)
        return noise

    # Cast an optional tensor to the DiT dtype.
    def apply_dtype_cast(self, value):
        if value is None:
            cast = None
        else:
            cast = value.to(dtype=self.dtype)
        return cast

    # Expand an optional tensor to batch size in DiT dtype.
    def apply_batch_cast(self, value, batch: int):
        if value is None:
            expanded = None
        else:
            expanded = value.expand(batch, *value.shape[1:])
            expanded = expanded.to(dtype=self.dtype)
        return expanded

    # Reverse-diffuse latent, matches AudioX sample_k.
    def denoise_diffusion_loop(
        self,
        noise_vector: torch.Tensor,
        input_condition: Conditioning,
        diffusion_step: int,
        noise_seeds: list[int] | None = None,
    ):
        sampling = self.spec.sampling

        cross_attn_cond = self.apply_dtype_cast(input_condition.text_timing)
        cross_attn_mask = self.apply_dtype_cast(input_condition.text_timing_mask)
        global_cond = self.apply_dtype_cast(input_condition.timing)
        negative_cond = self.apply_dtype_cast(input_condition.negative_text_timing)
        negative_mask = self.apply_dtype_cast(input_condition.negative_text_timing_mask)
        negative_global = self.apply_dtype_cast(input_condition.negative_timing)
        cond_inputs = DiffusionCondInputs(
            cross_attn_cond=cross_attn_cond,
            cross_attn_mask=cross_attn_mask,
            global_cond=global_cond,
            negative_cross_attn_cond=negative_cond,
            negative_cross_attn_mask=negative_mask,
            negative_global_cond=negative_global,
            cfg_scale=sampling.cfg_scale,
            rescale_cfg=sampling.rescale_cfg,
            batch_cfg=sampling.batch_cfg,
        )

        denoiser = VDenoiser(self.gen_model.model)

        sigmas = get_sigmas_polyexponential(
            diffusion_step,
            sampling.sigma_min,
            sampling.sigma_max,
            rho=sampling.rho,
            device=self.device,
        )
        x = noise_vector * sigmas[0]  # (B, D, F)

        sampler_args = {}
        if noise_seeds is not None:
            positive_sigmas = sigmas[sigmas > 0]
            sigma_low = positive_sigmas.min()
            sigma_high = sigmas.max()
            seed_list = []
            for seed_item in noise_seeds:
                seed_int = int(seed_item)
                seed_list.append(seed_int & 0x7FFFFFFF)
            sampler_args["noise_sampler"] = BrownianTreeNoiseSampler(
                x, sigma_low, sigma_high, seed=seed_list,
            )
        extra_args = vars(cond_inputs)
        denoised = sample_dpmpp_3m_sde(
            denoiser, x, sigmas, extra_args=extra_args,
            disable=noise_seeds is not None, **sampler_args,
        )
        return denoised

    # One v-prediction step to t=0: x0 = a*z_t - b*v.
    @torch.no_grad()
    def denoise_latent_step(
        self,
        latent_noisy: torch.Tensor,
        coeff_a: torch.Tensor,
        coeff_b: torch.Tensor,
        input_condition: Conditioning,
    ):
        batch = latent_noisy.shape[0]
        t = torch.atan2(coeff_b, coeff_a) * (2.0 / math.pi)  # (B,)
        t = t.to(device=latent_noisy.device)

        latent_cast = latent_noisy.to(dtype=self.dtype)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            cross_attn_cond = self.apply_batch_cast(input_condition.text_timing, batch)
            cross_attn_mask = self.apply_batch_cast(input_condition.text_timing_mask, batch)
            global_cond = self.apply_batch_cast(input_condition.timing, batch)
            v_pred = self.gen_model.model(
                latent_cast,
                t,
                cross_attn_cond=cross_attn_cond,
                cross_attn_mask=cross_attn_mask,
                global_cond=global_cond,
                cfg_scale=1.0,
            )
        shape = (batch, 1, 1)
        v_pred = v_pred.to(dtype=torch.float32)  # (B, D, F)
        coeff_a_view = coeff_a.view(shape)
        coeff_b_view = coeff_b.view(shape)
        latent_clean = coeff_a_view * latent_noisy - coeff_b_view * v_pred  # (B, D, F)
        return latent_clean

    # Sampler denoise from the level's sigma down to zero.
    @torch.no_grad()
    def denoise_latent_loop(self, latent_noisy: torch.Tensor, coeff_a: float, coeff_b: float,
                            input_condition: Conditioning, num_steps: int,
                            noise_seeds: list[int], cfg_scale: float = 1.0):
        sampling = self.spec.sampling
        if coeff_b == 0.0:
            latent_denoised = latent_noisy.to(dtype=torch.float32)
        else:
            batch = latent_noisy.shape[0]
            cond_inputs = {
                "cross_attn_cond": self.apply_batch_cast(input_condition.text_timing, batch),
                "cross_attn_mask": self.apply_batch_cast(input_condition.text_timing_mask, batch),
                "global_cond": self.apply_batch_cast(input_condition.timing, batch),
                "negative_cross_attn_cond": self.apply_batch_cast(
                    input_condition.negative_text_timing, batch),
                "negative_cross_attn_mask": self.apply_batch_cast(
                    input_condition.negative_text_timing_mask, batch),
                "negative_global_cond": self.apply_batch_cast(
                    input_condition.negative_timing, batch),
                "cfg_scale": cfg_scale,
                # the generate path rescales, so this loop must agree
                "rescale_cfg": sampling.rescale_cfg,
                "batch_cfg": sampling.batch_cfg,
            }
            if coeff_a > 0.0:
                sigma_start = min(coeff_b / coeff_a, sampling.sigma_max)
            else:
                sigma_start = sampling.sigma_max
            sigmas = get_schedule_sigmas(sigma_start, num_steps, self.device)
            # rescale so the first sigma matches the level exactly
            x = latent_noisy * (sigma_start / coeff_b)
            seeds = []
            for seed_item in noise_seeds:
                seed_int = int(seed_item)
                seeds.append(seed_int & 0x7FFFFFFF)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                denoiser = VDenoiser(self.gen_model.model)
                positive_sigmas = sigmas[sigmas > 0]
                sigma_low = positive_sigmas.min()
                sigma_high = sigmas.max()
                tree = BrownianTreeNoiseSampler(x, sigma_low, sigma_high, seed=seeds)
                x_start = x.to(dtype=self.dtype)
                denoised = sample_dpmpp_3m_sde(
                    denoiser, x_start, sigmas, extra_args=cond_inputs,
                    noise_sampler=tree, disable=True,
                )
            latent_denoised = denoised.to(dtype=torch.float32)  # (B, D, F)
        return latent_denoised

    # Generate batch-stable DiT latents without decoding.
    @torch.no_grad()
    def generate_latent(
        self,
        text_prompt: str | list[str],
        diffusion_step: int,
        seconds_start: float,
        seconds_total: float,
        seeds: list[int],
        latent_frames: int,
    ):
        if isinstance(text_prompt, str):
            prompts = [text_prompt]
        else:
            prompts = text_prompt
        if len(prompts) != len(seeds):
            raise ValueError("prompt and seed counts differ")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            noise_list = []
            for seed_item in seeds:
                seed_noise = self.sample_noise_vector(seed=seed_item, latent_frames=latent_frames)
                noise_list.append(seed_noise)
            noise = torch.cat(noise_list)  # (B, D, F)
            cond = self.make_conditioning(prompts, seconds_start, seconds_total)
            latent = self.denoise_diffusion_loop(noise, cond, diffusion_step, seeds)
        return latent

    # Decode from the vendored vae_sample latent space.
    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor):
        pretransform_fp32 = self.gen_model.pretransform.to(dtype=torch.float32)
        pretransform_eval = pretransform_fp32.eval()
        self.gen_model.pretransform = pretransform_eval
        latent = latent.to(dtype=torch.float32)
        # fp32 island, matching the upstream decode path
        with torch.amp.autocast(device_type="cuda", enabled=False):
            waveform = self.gen_model.pretransform.decode(latent)  # (B, C, T)
        return waveform

    # Text-to-audio generation pipeline.
    @torch.no_grad()
    def generate(
        self,
        text_prompt: str | list[str],
        diffusion_step: int,
        seconds_start: float,
        seconds_total: float,
        seed: int = -1,
    ):
        if isinstance(text_prompt, str):
            batch_size = 1
        else:
            batch_size = len(text_prompt)
        # only autocast in this wrapper; decode keeps fp32
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            noise = self.sample_noise_vector(batch_size=batch_size, seed=seed)
            cond = self.make_conditioning(text_prompt, seconds_start, seconds_total)
            latent = self.denoise_diffusion_loop(noise, cond, diffusion_step)
            # DiT trained on vendored vae_sample draws; decode that space
            audio = self.decode_latent(latent)
        window = int(seconds_total * self.spec.sample_rate)
        trim = min(window, audio.shape[-1])
        trimmed = audio[..., :trim]  # (B, C, T)
        return trimmed


# Tensors and gaps from one generator parity run.
@dataclass
class AudioXGeneratorParityResult:
    ours: torch.Tensor | None = None
    recomposed: torch.Tensor | None = None
    ours_latent: torch.Tensor | None = None
    ref_latent: torch.Tensor | None = None
    ours_matched: torch.Tensor | None = None
    ref: torch.Tensor | None = None
    ours_x0: torch.Tensor | None = None
    x0_ref: torch.Tensor | None = None
    wiring_diff: float = 0.0
    max_diff: float = 0.0
    step_diff: float = 0.0
    passed: bool = False


# Parity against generate_diffusion_cond and a VDenoiser step.
class AudioXGeneratorParity:
    def __init__(self, generator: AudioXGenerator, prompt: str, seed: int = 0,
                 steps: int = 50, seconds: int = 10, device: str = "cuda"):
        self.generator = generator
        self.prompt = prompt
        self.seed = seed
        self.steps = steps
        self.seconds = seconds
        self.device = device
        self.result = None

    # Read the stored prompt and build the generator.
    @classmethod
    def make_from_options(cls, ckpt_path: str = "", device: str = "cuda"):
        # prompt pairs with the stored parity clip
        prompt_file = registry.get_parity_prompt_path()
        prompt_path = Path(prompt_file)
        prompt_text = prompt_path.read_text(encoding="utf-8")
        prompt = prompt_text.strip()
        generator = AudioXGenerator(ckpt_path=ckpt_path, device=device)
        parity = cls(generator, prompt, device=device)
        return parity

    # Largest absolute elementwise gap between two tensors.
    def get_max_diff(self, ours: torch.Tensor, ref: torch.Tensor):
        gap = ours - ref
        gap_abs = gap.abs()
        gap_max = gap_abs.max()
        max_diff = gap_max.item()
        return max_diff

    # Zero video and audio prompts in upstream metadata form.
    def make_conditioning_metadata(self):
        model = self.generator
        conditioning = model.spec.conditioning
        frames = conditioning.video_fps * conditioning.cond_seconds
        samples = model.spec.sample_rate * conditioning.cond_seconds
        video_tensors = torch.zeros(
            1, frames, conditioning.video_channels, conditioning.video_size,
            conditioning.video_size, dtype=model.dtype,
        )
        audio_prompt = torch.zeros(1, model.spec.audio_channels, samples, dtype=model.dtype)
        conditioning_input = ConditioningInput(
            video_tensors=video_tensors,
            text_prompt=self.prompt,
            audio_prompt=audio_prompt,
        )
        metadata = conditioning_input.get_metadata()
        metadata_list = [metadata]
        return metadata_list

    # Decomposed rerun in bf16 must equal generate.
    def check_wiring(self, ours: torch.Tensor):
        model = self.generator
        # decomposed rerun in bf16: generate is pure wiring
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            noise = model.sample_noise_vector(batch_size=1, seed=self.seed)
            cond = model.make_conditioning(self.prompt, 0, self.seconds)
            latent_wired = model.denoise_diffusion_loop(noise, cond, self.steps)
            audio_wired = model.decode_latent(latent_wired)
        recomposed = audio_wired[..., : ours.shape[-1]]  # (1, C, T)
        wiring_diff = self.get_max_diff(ours, recomposed)
        print(f"generate wiring max_abs_diff={wiring_diff:.3e}")
        wiring_ok = torch.allclose(ours, recomposed, atol=1e-4, rtol=0)
        if not wiring_ok:
            print(f"FAIL: WIRING MISMATCH max_abs_diff={wiring_diff}")
        self.result.recomposed = recomposed
        self.result.wiring_diff = wiring_diff
        return wiring_ok

    # Our sampler and vendored generate_diffusion_cond, decoded alike.
    def check_diffusion(self, ours: torch.Tensor, metadata_list: list):
        model = self.generator
        sampling = model.spec.sampling
        # vendored sample_k autocasts fp16; parity arms match it
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            noise = model.sample_noise_vector(batch_size=1, seed=self.seed)
            cond = model.make_conditioning(self.prompt, 0, self.seconds)
            ours_latent = model.denoise_diffusion_loop(noise, cond, self.steps)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ref_latent = generate_diffusion_cond(
                model.gen_model,
                steps=self.steps,
                cfg_scale=sampling.cfg_scale,
                conditioning=metadata_list,
                sample_size=model.spec.sample_size,
                seed=self.seed,
                device=self.device,
                sampler_type=sampling.sampler_type,
                sigma_min=sampling.sigma_min,
                sigma_max=sampling.sigma_max,
                rho=sampling.rho,
                return_latents=True,
            )
        # same fp32 decode both arms, only diffusion paths compete
        ours_matched = model.decode_latent(ours_latent)
        ref = model.decode_latent(ref_latent)
        ref = ref.to(dtype=torch.float32)
        window = int(self.seconds * model.spec.sample_rate)
        expected_samples = min(window, ref.shape[-1])
        length_ok = ours.shape[-1] == expected_samples
        if not length_ok:
            print(
                f"FAIL: WRAPPER LENGTH MISMATCH ours={ours.shape[-1]} "
                f"expected={expected_samples}"
            )
        ref = ref[..., :expected_samples]  # (1, C, T)
        ours_matched = ours_matched[..., :expected_samples]
        shape_ok = ours.shape == ref.shape
        if not shape_ok:
            print(f"FAIL: SHAPE MISMATCH ours={tuple(ours.shape)} ref={tuple(ref.shape)}")

        max_diff = self.get_max_diff(ours_matched, ref)
        print(
            f"ours={tuple(ours_matched.shape)} ref={tuple(ref.shape)} "
            f"max_abs_diff={max_diff:.3e}"
        )
        # final comparison: wav [1, 2, T], atol 1e-4
        match_ok = torch.allclose(ours_matched, ref, atol=1e-4, rtol=0)
        if not match_ok:
            print(f"FAIL: MISMATCH max_abs_diff={max_diff}")
        diffusion_ok = length_ok and shape_ok and match_ok
        if diffusion_ok:
            print("PASS: decomposed generate matches vendored generate_diffusion_cond")
        self.result.ours_latent = ours_latent
        self.result.ref_latent = ref_latent
        self.result.ours_matched = ours_matched
        self.result.ref = ref
        self.result.max_diff = max_diff
        return diffusion_ok

    # Vendored VDenoiser x0 estimate for the same noisy latent.
    def denoise_ref_step(self, z_t: torch.Tensor, coeff_a: torch.Tensor,
                         coeff_b: torch.Tensor, cond: Conditioning):
        model = self.generator
        # this family has no global conditioning, so timing stays None
        if cond.timing is None:
            timing_cast = None
        else:
            timing_cast = cond.timing.to(dtype=model.dtype)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            step_denoiser = VDenoiser(model.gen_model.model)
            coeff_a_view = coeff_a.view(1, 1, 1)
            z_scaled = z_t / coeff_a_view
            z_input = z_scaled.to(dtype=model.dtype)
            sigma_ratio = coeff_b / coeff_a
            sigma = sigma_ratio.to(dtype=model.dtype)
            text_cast = cond.text_timing.to(dtype=model.dtype)
            mask_cast = cond.text_timing_mask.to(dtype=model.dtype)
            x0_ref = step_denoiser(
                z_input,
                sigma,
                cross_attn_cond=text_cast,
                cross_attn_mask=mask_cast,
                global_cond=timing_cast,
                cfg_scale=1.0,
            )
        x0_ref = x0_ref.to(dtype=torch.float32)  # (1, D, F)
        return x0_ref

    # One-step parity against VDenoiser at sigma 1.
    def check_step(self):
        model = self.generator
        coeff_value = 1.0 / math.sqrt(2.0)
        coeff_a = torch.full((1,), coeff_value, device=self.device)
        coeff_b = coeff_a.clone()
        torch.manual_seed(self.seed)
        z_ref = torch.randn(1, model.spec.latent_dim, 128, device=self.device)  # (1, D, F)
        coeff_a_view = coeff_a.view(1, 1, 1)
        z_signal = coeff_a_view * z_ref
        coeff_b_view = coeff_b.view(1, 1, 1)
        z_eps = torch.randn_like(z_ref)
        z_noise = coeff_b_view * z_eps
        z_t = z_signal + z_noise  # (1, D, F)
        cond = model.make_conditioning(self.prompt, 0, self.seconds)
        ours_x0 = model.denoise_latent_step(z_t, coeff_a, coeff_b, cond)
        x0_ref = self.denoise_ref_step(z_t, coeff_a, coeff_b, cond)
        step_diff = self.get_max_diff(ours_x0, x0_ref)
        print(f"one-step parity max_abs_diff={step_diff:.3e}")
        step_ok = torch.allclose(ours_x0, x0_ref, atol=3e-2, rtol=0)
        if step_ok:
            print("PASS: denoise_latent_step matches VDenoiser x0 estimate")
        else:
            print(f"FAIL: ONE-STEP MISMATCH max_abs_diff={step_diff}")
        self.result.ours_x0 = ours_x0
        self.result.x0_ref = x0_ref
        self.result.step_diff = step_diff
        return step_ok

    # Run the wiring, diffusion and one-step checks in order.
    def run(self):
        self.result = AudioXGeneratorParityResult()
        metadata_list = self.make_conditioning_metadata()
        ours = self.generator.generate(
            self.prompt, self.steps, seconds_start=0, seconds_total=self.seconds, seed=self.seed,
        )  # (1, C, T)
        self.result.ours = ours
        wiring_ok = self.check_wiring(ours)
        diffusion_ok = self.check_diffusion(ours, metadata_list)
        step_ok = self.check_step()
        self.result.passed = wiring_ok and diffusion_ok and step_ok
        result = self.result
        return result


@click.command()
@click.option("--data_root", type=Path, required=True)
@click.option("--ckpt_path", type=str, default="")
def main(data_root, ckpt_path):
    apply_data_root(data_root)
    parity = AudioXGeneratorParity.make_from_options(ckpt_path=ckpt_path)
    result = parity.run()
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
