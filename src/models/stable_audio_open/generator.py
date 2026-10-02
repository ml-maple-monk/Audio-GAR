from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import click
import torch
from k_diffusion.external import VDenoiser
from k_diffusion.sampling import (
    BrownianTreeNoiseSampler,
    get_sigmas_polyexponential,
    sample_dpmpp_2m_sde,
    sample_dpmpp_3m_sde,
)
from safetensors.torch import load_file
from torch import nn

from ...runtime.dataroot import apply_data_root
from ..generator import get_schedule_sigmas
from . import registry
from .vendor.generator_arch import (
    ConditionedDiffusionModelWrapper,
    DiTWrapper,
    create_multi_conditioner_from_conditioning_config,
    flash_attn_func,
    generate_diffusion_cond,
)
from .vendor.tokenizer_arch import (
    AudioAutoencoder,
    AutoencoderPretransform,
    OobleckDecoder,
    OobleckEncoder,
    VAEBottleneck,
)


SAMPLERS = {
    "dpmpp-2m-sde": sample_dpmpp_2m_sde,
    "dpmpp-3m-sde": sample_dpmpp_3m_sde,
}


@dataclass
class Conditioning:
    text_timing: torch.Tensor
    text_timing_mask: torch.Tensor
    timing: torch.Tensor
    negative_text_timing: torch.Tensor | None = None
    negative_text_timing_mask: torch.Tensor | None = None
    negative_timing: torch.Tensor | None = None


# L2 wrapper for Stable Audio Open.
class StableAudioGenerator(nn.Module):
    def __init__(self, name: str = "stable-audio-open", ckpt_path: str = "", device: str = "cuda",
                 determinism: bool = False, compile_blocks: bool = False):
        super().__init__()
        self.name = name

        self.spec = registry.spec(name)
        self.gen_model = self.make_gen_model()

        ckpt = ckpt_path or registry.get_ckpt_path(name)
        if ckpt:
            state_dict = load_file(ckpt)
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
        if compile_blocks:
            self.apply_compile()

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
            self.spec.dit.diffusion_objective,
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

    # T5 prompt conditioner plus the two timing-number conditioners.
    def make_conditioner(self):
        conditioning = self.spec.conditioning
        conditioning_dict = {
            "cond_dim": conditioning.cond_dim,
            "configs": [
                {
                    "id": "prompt",
                    "type": "t5",
                    "config": {
                        "t5_model_name": conditioning.t5_model_name,
                        "max_length": conditioning.max_length,
                    },
                },
                {
                    "id": "seconds_start",
                    "type": "number",
                    "config": {
                        "min_val": conditioning.seconds_min,
                        "max_val": conditioning.seconds_max,
                    },
                },
                {
                    "id": "seconds_total",
                    "type": "number",
                    "config": {
                        "min_val": conditioning.seconds_min,
                        "max_val": conditioning.seconds_max,
                    },
                },
            ],
        }
        conditioner = create_multi_conditioner_from_conditioning_config(conditioning_dict)
        return conditioner

    # Assemble the conditioned diffusion wrapper from its three parts.
    def make_gen_model(self):
        pretransform = self.make_pretransform()
        dit = self.make_dit()
        conditioner = self.make_conditioner()

        gen_model = ConditionedDiffusionModelWrapper(
            model=dit,
            conditioner=conditioner,
            pretransform=pretransform,
            io_channels=self.spec.latent_dim,
            sample_rate=self.spec.sample_rate,
            min_input_length=self.spec.downsampling_ratio,
            diffusion_objective=self.spec.dit.diffusion_objective,
            cross_attn_cond_ids=self.spec.conditioning.cross_attn_cond_ids,
            global_cond_ids=self.spec.conditioning.global_cond_ids,
        )
        return gen_model

    # Fail loud when the vendored flash attention path is absent.
    def check_flash_kernels(self, device):
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and flash_attn_func is None:
            raise Exception("flash_attn is not importable; the DiT requires it on cuda")

    # Resolve one declared sampler by name.
    def get_sampler(self, name: str):
        sampler = SAMPLERS[name]
        return sampler

    # Tf32 and autotuning, which determinism forbids.
    def apply_precision_policy(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Per block, since the whole model captures nothing.
    def apply_compile(self):
        blocks = None
        for module_item in self.gen_model.model.modules():
            layers = getattr(module_item, "layers", None)
            if isinstance(layers, nn.ModuleList) and len(layers) > 1:
                blocks = layers
                break
        if blocks is not None:
            n_blocks = len(blocks)
            for block_idx in range(n_blocks):
                block = blocks[block_idx]
                compiled_block = torch.compile(block, dynamic=False)
                blocks[block_idx] = compiled_block

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

    # Wrapper conditioning inputs for one prompt batch.
    def make_conditioning_inputs(self, batch: list[str], starts: list[float],
                                 totals: list[float], is_negative: bool):
        metadata = []
        for prompt_item, start_item, total_item in zip(batch, starts, totals, strict=True):
            metadata.append({
                "prompt": prompt_item,
                "seconds_start": start_item,
                "seconds_total": total_item,
            })
        per_source_embedding = self.gen_model.conditioner(metadata, self.device)
        inputs = self.gen_model.get_conditioning_inputs(per_source_embedding, negative=is_negative)
        return inputs

    # Build static conditioning once, reused across every denoise step.
    @torch.no_grad()
    def make_conditioning(
        self,
        text_prompt: str | list[str],
        seconds_start: float | list[float],
        seconds_total: float | list[float],
        negative_prompt: str | None = None,
    ):
        if isinstance(text_prompt, str):
            prompts = [text_prompt]
        else:
            prompts = text_prompt
        n_prompts = len(prompts)
        starts = seconds_start
        if isinstance(starts, (int, float)):
            start_value = float(starts)
            starts = [start_value] * n_prompts
        totals = seconds_total
        if isinstance(totals, (int, float)):
            total_value = float(totals)
            totals = [total_value] * n_prompts

        positive = self.make_conditioning_inputs(prompts, starts, totals, is_negative=False)
        cond = Conditioning(
            text_timing=positive["cross_attn_cond"],
            text_timing_mask=positive["cross_attn_mask"],
            timing=positive["global_cond"],
        )
        if negative_prompt is not None:
            negative_batch = [negative_prompt] * n_prompts
            negative = self.make_conditioning_inputs(negative_batch, starts, totals,
                                                     is_negative=True)
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

    # Expand an optional tensor to batch size in DiT dtype.
    def apply_batch_cast(self, value, batch: int):
        if value is None:
            expanded = None
        else:
            expanded = value.expand(batch, *value.shape[1:])
            expanded = expanded.to(dtype=self.dtype)
        return expanded

    # Reverse-diffuse latent with the declared sampler.
    def denoise_diffusion_loop(
        self,
        noise_vector: torch.Tensor,
        input_condition: Conditioning,
        diffusion_step: int,
        noise_seeds: list[int] | None = None,
    ):
        sampling = self.spec.sampling

        raw_inputs = {
            "cross_attn_cond": input_condition.text_timing,
            "cross_attn_mask": input_condition.text_timing_mask,
            "global_cond": input_condition.timing,
            "negative_cross_attn_cond": input_condition.negative_text_timing,
            "negative_cross_attn_mask": input_condition.negative_text_timing_mask,
            "negative_global_cond": input_condition.negative_timing,
        }
        cond_inputs = {}
        for key_item, value_item in raw_inputs.items():
            if value_item is not None:
                cond_inputs[key_item] = value_item.to(dtype=self.dtype)
            else:
                cond_inputs[key_item] = value_item
        cond_inputs.update(
            cfg_scale=sampling.cfg_scale, apg_scale=sampling.apg_scale,
            batch_cfg=sampling.batch_cfg, rescale_cfg=sampling.rescale_cfg,
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
        sampler = self.get_sampler(sampling.sampler_type)
        denoised = sampler(
            denoiser, x, sigmas, extra_args=cond_inputs,
            disable=noise_seeds is not None, **sampler_args,
        )
        return denoised

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
                # upstream defaults to full APG, so vanilla CFG is explicit
                "apg_scale": sampling.apg_scale,
                "batch_cfg": sampling.batch_cfg,
                "rescale_cfg": sampling.rescale_cfg,
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
                sampler = self.get_sampler(sampling.sampler_type)
                denoised = sampler(
                    denoiser, x_start, sigmas, extra_args=cond_inputs,
                    noise_sampler=tree, disable=True,
                )
            latent_denoised = denoised.to(dtype=torch.float32)  # (B, D, F)
        return latent_denoised

    # Decode from the vendored vae_sample latent space.
    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor):
        param_iter = self.gen_model.pretransform.parameters()
        first_param = next(param_iter)
        vae_dtype = first_param.dtype
        latent = latent.to(dtype=vae_dtype)
        # plain-dtype decode even under generate's outer autocast
        with torch.amp.autocast(device_type="cuda", enabled=False):
            audio = self.gen_model.pretransform.decode(latent)  # (B, C, T)
        return audio

    # Generate audio over the full window, trimmed to seconds_total.
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
        # only outer autocast; decode keeps a plain-dtype island
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            noise = self.sample_noise_vector(batch_size=batch_size, seed=seed)
            cond = self.make_conditioning(text_prompt, seconds_start, seconds_total)
            latent = self.denoise_diffusion_loop(noise, cond, diffusion_step)
            # DiT trained on vendored vae_sample draws; decode that space
            audio = self.decode_latent(latent)
        trim = int(seconds_total * self.spec.sample_rate)
        trimmed = audio[..., :trim]  # (B, C, T)
        return trimmed


# Tensors and gaps from one generator parity run.
@dataclass
class StableAudioGeneratorParityResult:
    ours: torch.Tensor | None = None
    ref_latent: torch.Tensor | None = None
    ref: torch.Tensor | None = None
    max_diff: float = 0.0
    passed: bool = False


# Parity against generate_diffusion_cond.
class StableAudioGeneratorParity:
    def __init__(self, generator: StableAudioGenerator, prompt: str, seed: int = 0,
                 steps: int = 100, seconds: int = 10, device: str = "cuda"):
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
        generator = StableAudioGenerator(
            name="stable-audio-open", ckpt_path=ckpt_path, device=device,
        )
        parity = cls(generator, prompt, device=device)
        return parity

    # Largest absolute elementwise gap between two tensors.
    def get_max_diff(self, ours: torch.Tensor, ref: torch.Tensor):
        gap = ours - ref
        gap_abs = gap.abs()
        gap_max = gap_abs.max()
        max_diff = gap_max.item()
        return max_diff

    # Prompt and timing in upstream metadata form.
    def make_conditioning_metadata(self):
        metadata = {"prompt": self.prompt, "seconds_start": 0, "seconds_total": self.seconds}
        metadata_list = [metadata]
        return metadata_list

    # Our wrapper and vendored generate_diffusion_cond, decoded alike.
    def check_diffusion(self, ours: torch.Tensor, metadata_list: list):
        model = self.generator
        sampling = model.spec.sampling
        # ref latents under the wrapper's autocast context
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
                apg_scale=sampling.apg_scale,
                batch_size=1,
                adapt_duration_to_conditioning=False,
                return_latents=True,
            )
        # same fp32 decode both arms, only diffusion paths compete
        ref = model.decode_latent(ref_latent)
        ref = ref.to(dtype=torch.float32)
        window = int(self.seconds * model.spec.sample_rate)
        expected_samples = min(window, ref.shape[-1])
        length_ok = ours.shape[-1] == expected_samples
        if not length_ok:
            print(f"FAIL: LENGTH MISMATCH ours={ours.shape[-1]} expected={expected_samples}")
        ref = ref[..., :expected_samples]  # (1, C, T)
        shape_ok = ours.shape == ref.shape
        if not shape_ok:
            print(f"FAIL: SHAPE MISMATCH ours={tuple(ours.shape)} ref={tuple(ref.shape)}")

        max_diff = self.get_max_diff(ours, ref)
        print(f"ours={tuple(ours.shape)} ref={tuple(ref.shape)} max_abs_diff={max_diff:.3e}")
        # final comparison: wav [1, 2, T], atol 1e-4
        match_ok = torch.allclose(ours, ref, atol=1e-4, rtol=0)
        if not match_ok:
            print(f"FAIL: MISMATCH max_abs_diff={max_diff}")
        diffusion_ok = length_ok and shape_ok and match_ok
        if diffusion_ok:
            print("PASS: wrapper output matches generate_diffusion_cond")
        self.result.ref_latent = ref_latent
        self.result.ref = ref
        self.result.max_diff = max_diff
        return diffusion_ok

    # Run the generate-versus-vendored check.
    def run(self):
        self.result = StableAudioGeneratorParityResult()
        metadata_list = self.make_conditioning_metadata()
        ours = self.generator.generate(
            self.prompt, self.steps, 0, self.seconds, seed=self.seed,
        )  # (1, C, T)
        self.result.ours = ours
        diffusion_ok = self.check_diffusion(ours, metadata_list)
        self.result.passed = diffusion_ok
        result = self.result
        return result


@click.command()
@click.option("--data_root", type=Path, required=True)
@click.option("--ckpt_path", type=str, default="")
def main(data_root, ckpt_path):
    apply_data_root(data_root)
    parity = StableAudioGeneratorParity.make_from_options(ckpt_path=ckpt_path)
    result = parity.run()
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
