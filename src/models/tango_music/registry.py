from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from ...dataset.registry import get_cosine_noise_coefficients
from ...runtime.dataroot import get_data_root

# finetune bundles: mel decoder plus vocoder, raw weights only
CODEC_TRAITS = {"vocoder_bundle": True, "decoder_target": "mel", "decoder_weights": ("raw",)}
# latent caching runs the denoiser in half precision by default
SPEED_DEFAULTS = {"gen_dtype": "float16"}


# Mel front end shared by the encoder and the vocoder.
@dataclass(frozen=True)
class SpecStft:
    filter_length: int = 1024
    hop_length: int = 160
    win_length: int = 1024
    n_mel_channels: int = 64
    sampling_rate: int = 16000
    mel_fmin: int = 0
    mel_fmax: int = 8000


# Mel autoencoder build params; encoder emits mean and logvar.
@dataclass(frozen=True)
class SpecVae:
    embed_dim: int = 8
    z_channels: int = 8
    double_z: bool = True
    channels: int = 128
    ch_mult: list[int] = field(default_factory=[1, 2, 4].copy)
    num_res_blocks: int = 2
    resolution: int = 256
    in_channels: int = 1
    out_channels: int = 1
    subband: int = 1
    dropout: float = 0.0
    mel_bins: int = 16
    scale: float = 0.9227914214134216


# Stable Diffusion 2 denoiser, reshaped for mel latents.
@dataclass(frozen=True)
class SpecUnet:
    in_channels: int = 8
    out_channels: int = 8
    sample_size: list[int] = field(default_factory=[32, 2].copy)
    block_out_channels: list[int] = field(default_factory=[320, 640, 1280, 1280].copy)
    attention_head_dim: list[int] = field(default_factory=[5, 10, 20, 20].copy)
    down_block_types: list[str] = field(
        default_factory=[
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ].copy
    )
    up_block_types: list[str] = field(
        default_factory=[
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ].copy
    )
    cross_attention_dim: int = 1024
    layers_per_block: int = 2
    norm_num_groups: int = 32
    norm_eps: float = 1e-05
    act_fn: str = "silu"
    downsample_padding: int = 1
    mid_block_scale_factor: int = 1
    use_linear_projection: bool = True
    upcast_attention: bool = True
    center_input_sample: bool = False
    flip_sin_to_cos: bool = True
    freq_shift: int = 0


# DDPM schedule the released checkpoint was trained under.
@dataclass(frozen=True)
class SpecScheduler:
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    prediction_type: str = "v_prediction"
    clip_sample: bool = False
    steps_offset: int = 1


# Cumulative alpha products of the scaled-linear schedule.
def make_scaled_linear_alphas(sched: SpecScheduler):
    span = sched.beta_end**0.5 - sched.beta_start**0.5
    stride = span / (sched.num_train_timesteps - 1)
    total = 1.0
    alpha_list = []
    for step_idx in range(sched.num_train_timesteps):
        beta = (sched.beta_start**0.5 + step_idx * stride) ** 2
        total *= 1.0 - beta
        alpha_list.append(total)
    return alpha_list


# Scaled-linear signal and noise weights at this level.
def get_linear_noise_coefficients(level: float):
    if not 0.0 <= level <= 1.0:
        raise ValueError(f"level must be in [0, 1], got {level}")
    if level == 0.0:
        coeffs = (1.0, 0.0)
    else:
        sched = SpecScheduler()
        alpha_list = make_scaled_linear_alphas(sched)
        n_alphas = len(alpha_list)
        start = round(level * (n_alphas - 1))
        alpha_start = alpha_list[start]
        coeffs = (alpha_start**0.5, (1.0 - alpha_start) ** 0.5)
    return coeffs


# Grid-suffix step count of one noise level.
def get_level_steps(num_steps: int, level: float, coeffs_fn=get_cosine_noise_coefficients):
    a_t, b_t = coeffs_fn(level)
    if b_t == 0.0:
        level_steps = 0
    else:
        norm = math.hypot(a_t, b_t)
        wanted = a_t / norm
        sched = SpecScheduler()
        alpha_list = make_scaled_linear_alphas(sched)
        # first index wins ties, as min() does
        start = 0
        best_gap = None
        for alpha_idx, alpha_item in enumerate(alpha_list):
            gap = abs(alpha_item**0.5 - wanted)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                start = alpha_idx
        ratio = sched.num_train_timesteps // num_steps
        grid_steps = (start - sched.steps_offset) // ratio + 1
        floored = max(1, grid_steps)
        level_steps = min(num_steps, floored)
    return level_steps


# Released Tango music protocol: 200 DDPM steps, guidance 3.
@dataclass(frozen=True)
class SpecSampling:
    steps: int = 200
    cfg_scale: float = 3.0
    sampler_type: str = "ddpm"
    sigma_min: float = 0.0
    sigma_max: float = 0.0
    rho: float = 0.0
    rescale_cfg: bool = False
    batch_cfg: bool = True


# Everything the tango music wrappers need to build one model.
@dataclass(frozen=True)
class SpecModel:
    sample_rate: int = 16000
    audio_channels: int = 1
    latent_channels: int = 8
    downsampling_ratio: int = 640
    generation_frames: int = 256
    stft: SpecStft = field(default_factory=SpecStft)
    vae: SpecVae = field(default_factory=SpecVae)
    unet: SpecUnet = field(default_factory=SpecUnet)
    scheduler: SpecScheduler = field(default_factory=SpecScheduler)
    sampling: SpecSampling = field(default_factory=SpecSampling)
    ckpt: str = "tango-music-af-ft-mc/pytorch_model_main.bin"
    vae_ckpt: str = "tango-music-af-ft-mc/pytorch_model_vae.bin"
    stft_ckpt: str = "tango-music-af-ft-mc/pytorch_model_stft.bin"
    text_encoder: str = "tango-music-af-ft-mc/flan-t5-large"
    bundle: str = "tango-music-af-ft-mc/bundle.json"
    weights_repo: str = "declare-lab/tango-music-af-ft-mc"
    weights_revision: str = "06019c8800144839172c353c857015ce5463224c"

    # Folded channel count the latent cache stores.
    @property
    def latent_dim(self):
        latent_dim = self.latent_channels * self.vae.mel_bins
        return latent_dim

    # Mel frames one generation decodes to.
    @property
    def mel_frames(self):
        mel_frames = self.generation_frames * 4
        return mel_frames

    # Autoencoder config in the upstream constructor's shape.
    def vae_config(self):
        ch_mult = list(self.vae.ch_mult)
        vae_kwargs = {
            "ddconfig": {
                "double_z": self.vae.double_z,
                "z_channels": self.vae.z_channels,
                "resolution": self.vae.resolution,
                "downsample_time": False,
                "in_channels": self.vae.in_channels,
                "out_ch": self.vae.out_channels,
                "ch": self.vae.channels,
                "ch_mult": ch_mult,
                "num_res_blocks": self.vae.num_res_blocks,
                "attn_resolutions": [],
                "dropout": self.vae.dropout,
            },
            "embed_dim": self.vae.embed_dim,
            "image_key": "fbank",
            "subband": self.vae.subband,
            "time_shuffle": 1,
            "scale_factor": self.vae.scale,
        }
        return vae_kwargs


MODELS: dict[str, SpecModel] = {
    "tango-music-af-ft-mc": SpecModel(),
}


BUNDLE_FIELDS = (
    "main", "vae", "stft", "text_encoder",
    "main_sha256", "vae_sha256", "stft_sha256", "text_encoder_sha256",
)


# Four tango music artifact paths, each paired with its sha256.
@dataclass(frozen=True)
class SpecBundle:
    main: str
    vae: str
    stft: str
    text_encoder: str
    main_sha256: str
    vae_sha256: str
    stft_sha256: str
    text_encoder_sha256: str


# Read a bundle manifest; check every artifact exists.
def load_bundle(path: str):
    manifest_path = Path(path)
    manifest_text = manifest_path.read_text()
    data = json.loads(manifest_text)
    root = get_data_root()
    for key_item in ("main", "vae", "stft", "text_encoder"):
        artifact_path = root / data[key_item]
        if not artifact_path.exists():
            raise ValueError(f"bundle manifest {path} names missing artifact: {data[key_item]}")
    field_dict = {}
    for key_item in BUNDLE_FIELDS:
        field_dict[key_item] = data[key_item]
    bundle = SpecBundle(**field_dict)
    return bundle


# Typed build and sampling spec for a model name.
def spec(name: str):
    model_spec = MODELS[name]
    return model_spec


# Native waveform sample rate of one model.
def get_native_sample_rate(name: str):
    model_spec = MODELS[name]
    return model_spec.sample_rate


# Absolute bundle manifest path under the data root.
def get_ckpt_path(name: str):
    rel = MODELS[name].bundle
    if rel:
        root = get_data_root()
        ckpt_path = str(root / rel)
    else:
        ckpt_path = None
    return ckpt_path


# Absolute autoencoder checkpoint path under the data root.
def get_vae_ckpt_path(name: str):
    rel = MODELS[name].vae_ckpt
    if rel:
        root = get_data_root()
        ckpt_path = str(root / rel)
    else:
        ckpt_path = None
    return ckpt_path


# Absolute mel front end checkpoint path under the data root.
def get_stft_ckpt_path(name: str):
    rel = MODELS[name].stft_ckpt
    if rel:
        root = get_data_root()
        ckpt_path = str(root / rel)
    else:
        ckpt_path = None
    return ckpt_path


# Absolute staged text encoder directory under the data root.
def get_text_encoder_path(name: str):
    root = get_data_root()
    rel = MODELS[name].text_encoder
    encoder_path = str(root / rel)
    return encoder_path
