from dataclasses import dataclass
from functools import partial
from importlib import import_module

import torch
from torch import nn

from ..dataset.registry import get_cosine_noise_coefficients


# Tokenizer class for the family whose prefix matches the name.
def get_tokenizer_class(tokenizer_name: str):
    # families import lazily, so a broken family costs nothing
    family_cls = None
    is_tango_only = not tokenizer_name.startswith(("tango2", "tango-music"))
    if tokenizer_name.startswith("stable-audio-open"):
        from .stable_audio_open.tokenizer import StableAudioTokenizer
        family_cls = StableAudioTokenizer
    if tokenizer_name.startswith("audiox"):
        from .audioX.tokenizer import AudioXTokenizer
        family_cls = AudioXTokenizer
    if tokenizer_name.startswith(("audioldm2", "audioldm1")):
        from .audioldm2.tokenizer import AudioLDM2Tokenizer
        family_cls = AudioLDM2Tokenizer
    if tokenizer_name.startswith("tango2"):
        from .tango2.tokenizer import Tango2Tokenizer
        family_cls = Tango2Tokenizer
    if tokenizer_name.startswith("tango-music"):
        from .tango_music.tokenizer import TangoMusicTokenizer
        family_cls = TangoMusicTokenizer
    if tokenizer_name.startswith("tango") and is_tango_only:
        from .tango2.tokenizer import Tango2Tokenizer
        family_cls = Tango2Tokenizer
    return family_cls


# Pretrained path from the family owning the name.
def get_family_ckpt_path(tokenizer_name: str):
    family_cls = get_tokenizer_class(tokenizer_name)
    module = import_module(family_cls.__module__)
    ckpt_path = module.registry.get_ckpt_path(tokenizer_name)
    return ckpt_path


@dataclass(frozen=True)
class SpecCodecTraits:
    vocoder_bundle: bool = False
    decoder_target: str = "decoder"
    decoder_weights: tuple[str, ...] = ("ema", "raw")


# Family bundle traits; defaults when the family names none.
def get_codec_traits(tokenizer_name: str):
    family_cls = get_tokenizer_class(tokenizer_name)
    module = import_module(family_cls.__module__)
    trait_dict = getattr(module.registry, "CODEC_TRAITS", {})
    traits = SpecCodecTraits(**trait_dict)
    return traits


class HookCapture:
    def __init__(self, layer_paths: list[str], reduce=None):
        self.reduce = reduce
        self.captured = {}
        for path_item in layer_paths:
            self.captured[path_item] = []

    # Store one module output under its dotted path.
    def hook_output(self, path: str, module, inputs, output):
        out = output
        if isinstance(output, (tuple, list)):
            out = output[0]
        if self.reduce is None:
            capture = out.detach()
        else:
            capture = self.reduce(out)
        self.captured[path].append(capture)

    # Paths whose module never fired during decode.
    def get_silent_paths(self):
        silent_list = []
        for path_item, tensor_list in self.captured.items():
            if not tensor_list:
                silent_list.append(path_item)
        return silent_list


class AudioTokenizer(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    # Native sample rate of the codec.
    @property
    def sample_rate(self):
        rate = self.model.sample_rate
        return rate

    # Latent frames per second.
    @property
    def frame_rate(self):
        rate = self.model.frame_rate
        return rate

    # Audio channels the codec reads and writes.
    @property
    def audio_channels(self):
        channels = getattr(self.model, "audio_channels", 1)
        return channels

    # Device the codec weights sit on.
    @property
    def device(self):
        device = self.model.device
        return device

    # Waveform to the family's encoder output.
    def encode(self, wav: torch.Tensor):
        encoded = self.model.encode(wav)
        return encoded

    # Continuous-latent families only, posterior draw.
    def encode_latent_sample(self, wav: torch.Tensor, eps: torch.Tensor):
        latent = self.model.encode_latent_sample(wav, eps)
        return latent

    # Codes back to waveform.
    def decode(self, codes: torch.Tensor):
        wav = self.model.decode(codes)  # (B, C, T)
        return wav

    # Encode then decode one waveform batch.
    def reconstruct(self, wav: torch.Tensor):
        recon = self.model.reconstruct(wav)  # (B, C, T)
        return recon

    # Dotted submodule paths, for picking capture layers.
    def get_module_paths(self):
        path_list = []
        for name_item, module_item in self.model.named_modules():
            if name_item:
                path_list.append(name_item)
        return path_list

    # Full autoencoder module, continuous-latent families only.
    def get_vae_module(self):
        wrapped = getattr(self.model, "pretransform", None)
        vae_module = wrapped.model
        return vae_module

    # Checkpoint loaded by the family tokenizer.
    def get_weights_path(self):
        path = getattr(self.model, "weights_path", "")
        if not path:
            raise ValueError(f"{self.model.name}: family loaded no pretrained weights")
        return path

    # Latent-space rescale; identity when the family has none.
    def apply_latent_scale(self, latent: torch.Tensor, direction: str):
        scale_fn = getattr(self.model, "apply_latent_scale", None)
        if scale_fn is None:
            scaled = latent
        else:
            scaled = scale_fn(latent, direction)
        return scaled

    # Encoder output as decode() accepts it.
    @torch.no_grad()
    def get_codes_for_decode(self, wav: torch.Tensor):
        encoded = self.model.encode(wav)
        sample_fn = getattr(self.model, "vae_sample", None)
        if sample_fn is None:
            codes = encoded
        else:
            codes = sample_fn(encoded)
        return codes

    # Decode under forward hooks; captures listed per module path.
    @torch.no_grad()
    def decode_with_hidden_states(self, codes: torch.Tensor, layer_paths: list[str], reduce=None):
        module_dict = dict(self.model.named_modules())
        capture = HookCapture(layer_paths, reduce)
        handle_list = []
        for path_item in layer_paths:
            hook_fn = partial(capture.hook_output, path_item)
            handle = module_dict[path_item].register_forward_hook(hook_fn)
            handle_list.append(handle)
        try:
            wav = self.decode(codes)
        finally:
            for handle_item in handle_list:
                handle_item.remove()
        silent_list = capture.get_silent_paths()
        if silent_list:
            raise RuntimeError(f"decode never ran modules {silent_list}; wrong path for this family?")
        return wav, capture.captured

    # Decode a_t*z + b_t*eps; one eps per sweep.
    @torch.no_grad()
    def noisy_reconstruction(self, wav: torch.Tensor, noise_levels, seed: int | None = None):
        scalar = isinstance(noise_levels, (int, float))
        level_list = []
        if scalar:
            level_list.append(float(noise_levels))
        else:
            for level_item in noise_levels:
                level_list.append(float(level_item))
        z = self.model.encode_latent(wav)
        if seed is None:
            eps = torch.randn_like(z)
        else:
            generator = torch.Generator(device=z.device)
            generator = generator.manual_seed(seed)
            eps = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)
        out_list = []
        for level_item in level_list:
            a_t, b_t = get_cosine_noise_coefficients(level_item)
            noisy = a_t * z + b_t * eps
            decoded = self.model.decode_latent(noisy)
            clamped = decoded.clamp_(-1, 1)
            out_list.append(clamped)
        if scalar:
            result = out_list[0]
        else:
            result = out_list
        return result


# Build one family tokenizer, family picked by name.
def load_tokenizer(tokenizer_name: str, ckpt_path: str = "", device: str = "cuda", **kwargs):
    family_cls = get_tokenizer_class(tokenizer_name)
    family_model = family_cls(name=tokenizer_name, ckpt_path=ckpt_path, device=device, **kwargs)
    tokenizer = AudioTokenizer(family_model)
    return tokenizer
