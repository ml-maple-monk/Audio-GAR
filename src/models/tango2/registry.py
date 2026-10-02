from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

from ...dataset.registry import get_cosine_noise_coefficients
from ...runtime.dataroot import get_data_root

# finetune bundles: mel decoder plus vocoder, raw weights only
CODEC_TRAITS = {"vocoder_bundle": True, "decoder_target": "mel", "decoder_weights": ("raw",)}


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


# Released Tango 2 protocol: 200 DDPM steps, guidance 3.
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


# Everything the tango2 wrappers need to build one model.
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
    ckpt: str = "tango2-full/pytorch_model_main.bin"
    vae_ckpt: str = "tango2-full/pytorch_model_vae.bin"
    stft_ckpt: str = "tango2-full/pytorch_model_stft.bin"
    text_encoder: str = "tango2-full/flan-t5-large"
    bundle: str = "tango2-full/bundle.json"
    weights_repo: str = "declare-lab/tango2-full"
    weights_revision: str = "b779f5a77c18e21b0c093bf4700e3ad743f6ef0b"

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
    "tango2-full": SpecModel(),
    # paper checkpoint; vae, stft and t5 bytes match tango2-full
    "tango2": SpecModel(
        ckpt="tango2/pytorch_model_main.bin",
        bundle="tango2/bundle.json",
        weights_repo="declare-lab/tango2",
        weights_revision="86d7260a578b8a51218d26f1ba4e8398eaac9761",
    ),
    # tango1 original release; vae and stft match tango2-full
    "tango": SpecModel(
        ckpt="tango/pytorch_model_main.bin",
        stft_ckpt="tango/pytorch_model_stft.bin",
        bundle="tango/bundle.json",
        weights_repo="declare-lab/tango",
        weights_revision="5ed94b368f5a05993d20938dad1d5fbb832c4f7f",
    ),
    # tango1 full-data release; vae and stft match tango2-full
    "tango-full": SpecModel(
        ckpt="tango-full/pytorch_model_main.bin",
        stft_ckpt="tango-full/pytorch_model_stft.bin",
        bundle="tango-full/bundle.json",
        weights_repo="declare-lab/tango-full",
        weights_revision="28c2cd4451d772cf3f2ae4927669ad91e938ebca",
    ),
    # tango-full finetuned on audiocaps
    "tango-full-ft-audiocaps": SpecModel(
        ckpt="tango-full-ft-audiocaps/pytorch_model_main.bin",
        stft_ckpt="tango-full-ft-audiocaps/pytorch_model_stft.bin",
        bundle="tango-full-ft-audiocaps/bundle.json",
        weights_repo="declare-lab/tango-full-ft-audiocaps",
        weights_revision="bcdef43a7759667f436cc7444c2f5e7498654c60",
    ),
    # tango-full finetuned on audiocaps plus musiccaps
    "tango-full-ft-audio-music-caps": SpecModel(
        ckpt="tango-full-ft-audio-music-caps/pytorch_model_main.bin",
        stft_ckpt="tango-full-ft-audio-music-caps/pytorch_model_stft.bin",
        bundle="tango-full-ft-audio-music-caps/bundle.json",
        weights_repo="declare-lab/tango-full-ft-audio-music-caps",
        weights_revision="9ac44faf2e7e1e8361fb0f56a5dd2eb60b4bcb08",
    ),
    # af-audioset base finetuned on audiocaps; vae matches tango-music-af-ft-mc
    "tango-af-ac-ft-ac": SpecModel(
        ckpt="tango-af-ac-ft-ac/pytorch_model_main.bin",
        vae_ckpt="tango-music-af-ft-mc/pytorch_model_vae.bin",
        stft_ckpt="tango-af-ac-ft-ac/pytorch_model_stft.bin",
        bundle="tango-af-ac-ft-ac/bundle.json",
        weights_repo="declare-lab/tango-af-ac-ft-ac",
        weights_revision="bd88441c43484708c10af30784fb7b736295a168",
    ),
}


BUNDLE_FIELDS = (
    "main", "vae", "stft", "text_encoder",
    "main_sha256", "vae_sha256", "stft_sha256", "text_encoder_sha256",
)


# Four tango2 artifact paths, each paired with its sha256.
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


# AudioLDM autoencoder loss values, KL and LPIPS dropped.
@dataclass(frozen=True)
class SpecMelVaeLoss:
    logvar_init: float = 0.0
    disc_num_layers: int = 3
    disc_in_channels: int = 1
    disc_filters: int = 64
    disc_factor: float = 1.0
    disc_weight: float = 0.5
    disc_start: int = 0
    adaptive_clamp: float = 1e4
    adaptive_eps: float = 1e-4


# Two Adam optimizers, released config, linear decay.
@dataclass(frozen=True)
class SpecMelVaeOptim:
    lr: float = 8e-6
    betas: tuple[float, float] = (0.5, 0.9)
    lr_decay: str = "linear"


# jik876 generator recipe values at the release defaults.
@dataclass(frozen=True)
class SpecVocoderLoss:
    mel_weight: float = 45.0
    periods: tuple[int, ...] = (2, 3, 5, 7, 11)
    msd_scales: int = 3


# AdamW pair with the exponential decay mapping.
@dataclass(frozen=True)
class SpecVocoderOptim:
    lr: float = 2e-4
    betas: tuple[float, float] = (0.8, 0.99)
    lr_decay: float = 0.999
    weight_decay: float = 0.01
    # zero means one split pass, resolved at the entry
    lr_decay_every: int = 0
    # HiFiGAN exponential per split pass, or linear to zero
    lr_schedule: str = "exponential"


# Batch values and run controls both lanes share.
@dataclass(frozen=True)
class SpecCodecFit:
    batch_clips: int = 4
    accumulation: int = 1
    crop_frames: int = 13
    steps: int = 5000
    checkpoint_every: int = 5000
    logging_every: int = 10
    eval_every: int = 1_000_000
    levels: tuple[str, ...] = ("ref",)
    init_seed_index: int = -1
    peak_norm: bool = False


# Loader controls outside the slot hash, plus precision policy.
@dataclass(frozen=True)
class SpecCodecRuntime:
    data_workers: int = 8
    data_prefetch: int = 4
    autocast_dtype: str = "bfloat16"
    allow_tf32: bool = True
    grad_clip: str = "none"


# Upstream pins and divergences accepted on purpose.
@dataclass(frozen=True)
class SpecMelVaeSource:
    train_repo_commit: str = "702a638d023b008a2d9a45cdf1e1f4fcdc590dfc"
    taming_commit: str = "3ba01b241669f5ade541ce990f7650a3b8f65318"
    upstream_disc_start: int = 50001
    upstream_lr_paper: float = 4.5e-6
    lr_basis: str = "released 16k_64.yaml over the paper"
    kl_status: str = "dropped, encoder frozen"
    discriminator_weight_status: str = "zenodo 14342967 lightning ckpt, step 890000"
    disc_ckpt: str = "audioldm_vae_16k/vae_mel_16k_64bins.ckpt"
    disc_sha256: str = "d1879044a35c3c38600c70f2e70bee8f28706982af51be6dd7e00ec2e6788807"
    lpips_status: str = "dropped, the paper reports no perceptual term"


# Upstream pins and divergences accepted on purpose.
@dataclass(frozen=True)
class SpecVocoderSource:
    train_repo_commit: str = "4769534d45265d52a904b850da5a622601885777"
    config_name: str = "HIFIGAN_16K_64"
    upstream_segment_samples: int = 8192
    segment_basis: str = "13 latent frames, closest on-grid above 8192"
    mel_input: str = "decoded from cached latents, not ground truth"
    discriminator_weight_status: str = "never released, seeded fresh"


DEFAULT_RECIPE = "audioldm_released"


# Decoder-side mel autoencoder finetune recipe.
@dataclass(frozen=True)
class SpecMelVaeTraining:
    model_name: str = "tango2"
    recipe: str = DEFAULT_RECIPE
    loss: SpecMelVaeLoss = field(default_factory=SpecMelVaeLoss)
    optim: SpecMelVaeOptim = field(default_factory=SpecMelVaeOptim)
    fit: SpecCodecFit = field(default_factory=SpecCodecFit)
    runtime: SpecCodecRuntime = field(default_factory=SpecCodecRuntime)
    source: SpecMelVaeSource = field(default_factory=SpecMelVaeSource)


# HiFi-GAN generator finetune recipe.
@dataclass(frozen=True)
class SpecVocoderTraining:
    model_name: str = "tango2"
    recipe: str = DEFAULT_RECIPE
    loss: SpecVocoderLoss = field(default_factory=SpecVocoderLoss)
    optim: SpecVocoderOptim = field(default_factory=SpecVocoderOptim)
    fit: SpecCodecFit = field(default_factory=SpecCodecFit)
    runtime: SpecCodecRuntime = field(default_factory=SpecCodecRuntime)
    source: SpecVocoderSource = field(default_factory=SpecVocoderSource)


# released 8e-6/(0.5,0.9); disc_start 0 for finetune, upstream pin in source
VAE_RELEASED_FIT = SpecCodecFit(crop_frames=250)
VAE_RELEASED = SpecMelVaeTraining(fit=VAE_RELEASED_FIT)
# published paper LR and batch; mixup stays an unimplemented divergence
VAE_PAPER_OPTIM = SpecMelVaeOptim(lr=4.5e-6)
VAE_PAPER_FIT = replace(VAE_RELEASED.fit, batch_clips=6)
VAE_PAPER = replace(
    VAE_RELEASED,
    recipe="audioldm_paper",
    optim=VAE_PAPER_OPTIM,
    fit=VAE_PAPER_FIT,
)
# today's pilot crop, kept live as its own named profile
VAE_PILOT_FIT = SpecCodecFit(crop_frames=32)
VAE_PILOT = SpecMelVaeTraining(recipe="finetune_pilot", fit=VAE_PILOT_FIT)

# published AdamW pair; batch 16 is the per-device split
VOCODER_RELEASED_FIT = SpecCodecFit(batch_clips=16)
VOCODER_RELEASED = SpecVocoderTraining(fit=VOCODER_RELEASED_FIT)
VOCODER_PAPER = replace(VOCODER_RELEASED, recipe="audioldm_paper")
VOCODER_PILOT = SpecVocoderTraining(recipe="finetune_pilot")

RECIPES: dict[str, tuple[SpecMelVaeTraining, SpecVocoderTraining]] = {
    "audioldm_released": (VAE_RELEASED, VOCODER_RELEASED),
    "audioldm_paper": (VAE_PAPER, VOCODER_PAPER),
    "finetune_pilot": (VAE_PILOT, VOCODER_PILOT),
}


# Joint lane vocoder windows, levels and grad-norm cadence.
@dataclass(frozen=True)
class SpecJointWindow:
    # measured: 32 mel frames of context erase the cut-edge error
    margin_frames: int = 8
    # windows 40 dB under the clip peak carry no level
    silence_peak: float = 0.005
    peak: float = 0.5
    # multiple of the default logging_every, so readings are kept
    grad_norm_every: int = 50


# Weight-name prefixes each named train scope releases.
@dataclass(frozen=True)
class SpecJointScopes:
    decoder_first: tuple[str, ...] = ("post_quant_conv.", "decoder.conv_in.")
    vocoder_last: tuple[str, ...] = ("conv_post.",)


# Joint CLI values; None fields follow the chosen recipe.
@dataclass(frozen=True)
class TrainJointConfig:
    cache_dirs: str
    data_root: Path
    out_dir: Path
    recipe: str = DEFAULT_RECIPE
    level: str | None = None
    max_train_steps: int | None = None
    per_device_train_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    learning_rate: float | None = None
    vocoder_learning_rate: float | None = None
    vocoder_loss_weight: float = 100.0
    vocoder_crop_frames: int = 13
    vocoder_windows_per_clip: int = 4
    vocoder_lr_schedule: str = "linear"
    decoder_train: str = "all"
    vocoder_train: str = "all"
    vocoder_init: str = "pretrained"
    critics_only: bool = False
    crop_frames: int | None = None
    checkpointing_steps: int | None = None
    logging_steps: int | None = None
    init_from: str = ""
    init_critics_from: str = ""
    data_workers: int | None = None
    data_prefetch: int | None = None
    precision: str = "bf16"
    seed: int = 42
    device: str = "cuda"


# Named decoder-side recipe, bound to one tango codec.
def get_mel_vae_training(name: str = "tango2", recipe: str = DEFAULT_RECIPE):
    recipe_pair = RECIPES[recipe]
    training = replace(recipe_pair[0], model_name=name)
    return training


# Named vocoder recipe, bound to one tango codec.
def get_vocoder_training(name: str = "tango2", recipe: str = DEFAULT_RECIPE):
    recipe_pair = RECIPES[recipe]
    training = replace(recipe_pair[1], model_name=name)
    return training


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
