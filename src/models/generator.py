import math
from dataclasses import replace
from functools import partial
from importlib import import_module

import torch
from torch import nn

from ..dataset.registry import get_cosine_noise_coefficients, get_schedule_steps


# Sigma grid uniform in t, level down to zero.
def get_schedule_sigmas(sigma_start: float, num_steps: int, device):
    t_start = math.atan(sigma_start) * 2.0 / math.pi
    grid = torch.linspace(t_start, 0.0, num_steps + 1, dtype=torch.float64, device=device)
    tan_grid = torch.tan(grid * (math.pi / 2))
    sigmas = tan_grid.clamp_(0.0, sigma_start)
    # endpoints analytically, the tan roundtrip loses precision
    sigmas[0] = sigma_start
    sigmas[-1] = 0.0
    # float32 keeps the solver's log sigma ratios exact
    sigmas_f32 = sigmas.float()
    return sigmas_f32


GRID_SUFFIX_PREFIXES = ("tango",)


# Generator class for the family whose prefix matches the name.
def get_generator_class(generator_name: str):
    # families import lazily, so a broken family costs nothing
    family_cls = None
    is_tango_only = not generator_name.startswith(("tango2", "tango-music"))
    if generator_name.startswith("stable-audio-open"):
        from .stable_audio_open.generator import StableAudioGenerator
        family_cls = StableAudioGenerator
    if generator_name.startswith("audiox"):
        from .audioX.generator import AudioXGenerator
        family_cls = AudioXGenerator
    if generator_name.startswith(("audioldm2", "audioldm1")):
        from .audioldm2.generator import AudioLDM2Generator
        family_cls = AudioLDM2Generator
    if generator_name.startswith("tango2"):
        from .tango2.generator import Tango2Generator
        family_cls = Tango2Generator
    if generator_name.startswith("tango-music"):
        from .tango_music.generator import TangoMusicGenerator
        family_cls = TangoMusicGenerator
    if generator_name.startswith("tango") and is_tango_only:
        from .tango2.generator import Tango2Generator
        family_cls = Tango2Generator
    return family_cls


# Slot label of the truncation rule this family runs.
def get_steps_mode(generator_name: str):
    if generator_name.startswith(GRID_SUFFIX_PREFIXES):
        mode = "grid_suffix"
    else:
        mode = "t_scaled"
    return mode


# Registry module of the tango family this name selects.
def get_tango_registry(generator_name: str):
    if generator_name.startswith("tango-music"):
        from .tango_music import registry
    else:
        from .tango2 import registry
    return registry


# Level to signal and noise weights under this schedule.
def get_noise_coefficients_fn(generator_name: str, schedule: str = "cosine"):
    is_tango = generator_name.startswith(GRID_SUFFIX_PREFIXES)
    # both guards stop a silent fall to the wrong schedule
    if schedule not in ("cosine", "scaled_linear"):
        raise ValueError(f"unknown noise schedule {schedule}")
    if schedule == "scaled_linear" and not is_tango:
        raise ValueError(f"{schedule} schedule needs a tango generator, got {generator_name}")
    if schedule == "cosine":
        coeffs_fn = get_cosine_noise_coefficients
    else:
        tango_registry = get_tango_registry(generator_name)
        coeffs_fn = tango_registry.get_linear_noise_coefficients
    return coeffs_fn


# Per-level step counter matching the family's truncation.
def get_level_steps_fn(generator_name: str, schedule: str = "cosine"):
    coeffs_fn = get_noise_coefficients_fn(generator_name, schedule)
    if generator_name.startswith(GRID_SUFFIX_PREFIXES):
        tango_registry = get_tango_registry(generator_name)
        steps_fn = partial(tango_registry.get_level_steps, coeffs_fn=coeffs_fn)
    else:
        steps_fn = get_schedule_steps
    return steps_fn


# Registry module of the family owning the name.
def get_generator_registry(generator_name: str):
    module_name = None
    is_tango_only = not generator_name.startswith(("tango2", "tango-music"))
    if generator_name.startswith("audiox"):
        module_name = ".audioX"
    if generator_name.startswith("stable-audio-open"):
        module_name = ".stable_audio_open"
    if generator_name.startswith("tango2"):
        module_name = ".tango2"
    if generator_name.startswith("tango-music"):
        module_name = ".tango_music"
    if generator_name.startswith(("audioldm2", "audioldm1")):
        module_name = ".audioldm2"
    if generator_name.startswith("tango") and is_tango_only:
        module_name = ".tango2"
    registry = import_module(module_name + ".registry", __package__)
    return registry


# Family model spec from its registry, no model load.
def get_generator_spec(generator_name: str):
    registry = get_generator_registry(generator_name)
    spec = registry.spec(generator_name)
    return spec


# Family inference step count from its registry, no model load.
def get_generator_steps(generator_name: str):
    spec = get_generator_spec(generator_name)
    steps = spec.sampling.steps
    return steps


# Cache speed defaults, or None when the family has none.
def get_speed_defaults(generator_name: str):
    registry = get_generator_registry(generator_name)
    speed_defaults = getattr(registry, "SPEED_DEFAULTS", None)
    return speed_defaults


# Family release sampling spec, no model load.
def get_generator_sampling(generator_name: str):
    spec = get_generator_spec(generator_name)
    sampling = spec.sampling
    return sampling


class AudioGenerator(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    # Typed build and sampling spec of the loaded family.
    @property
    def spec(self):
        spec = self.model.spec
        return spec

    # Pin the loaded spec to the protocol sampler settings.
    def apply_sampling(self, steps: int, cfg_scale: float):
        sampling = replace(self.model.spec.sampling, steps=steps, cfg_scale=cfg_scale)
        self.model.spec = replace(self.model.spec, sampling=sampling)
        pinned = self.model.spec.sampling
        return pinned

    # Forward verbatim; call signatures differ between families.
    def generate(self, *args, **kwargs):
        audio = self.model.generate(*args, **kwargs)
        return audio

    # Diffusion families return latents before codec decoding.
    def generate_latent(
        self,
        text_prompt: str | list[str],
        diffusion_step: int,
        seconds_start: float,
        seconds_total: float,
        seeds: list[int],
        latent_frames: int,
    ):
        latent = self.model.generate_latent(
            text_prompt, diffusion_step, seconds_start, seconds_total, seeds, latent_frames,
        )
        return latent

    # Family conditioning for a prompt batch.
    def make_conditioning(self, *args, **kwargs):
        cond = self.model.make_conditioning(*args, **kwargs)
        return cond

    # Diffusion families only, one v-step to t=0.
    def denoise_latent_step(self, *args, **kwargs):
        latent = self.model.denoise_latent_step(*args, **kwargs)
        return latent

    # Diffusion families only, sampler loop to t=0.
    def denoise_latent_loop(self, *args, **kwargs):
        latent = self.model.denoise_latent_loop(*args, **kwargs)
        return latent


# Build one family generator, family picked by name.
def load_generator(generator_name: str, ckpt_path: str = "", device: str = "cuda", **kwargs):
    family_cls = get_generator_class(generator_name)
    family_model = family_cls(name=generator_name, ckpt_path=ckpt_path, device=device, **kwargs)
    generator = AudioGenerator(family_model)
    return generator
