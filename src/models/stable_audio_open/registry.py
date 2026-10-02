from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ...runtime.dataroot import get_data_root


# Oobleck VAE params; encoder emits mean+scale, so 2*latent_dim.
@dataclass(frozen=True)
class VaeConfig:
    channels: int = 128
    c_mults: list[int] = field(default_factory=[1, 2, 4, 8, 16].copy)
    strides: list[int] = field(default_factory=[2, 4, 4, 8, 8].copy)
    use_snake: bool = True
    final_tanh: bool = False
    scale: float = 1.0
    iterate_batch: bool = True


# DiT denoiser build params.
@dataclass(frozen=True)
class SpecDit:
    embed_dim: int = 1536
    depth: int = 24
    num_heads: int = 24
    cond_token_dim: int = 768
    global_cond_dim: int = 1536
    project_cond_tokens: bool = False
    transformer_type: str = "continuous_transformer"
    diffusion_objective: str = "v"


# T5 prompt + two timing NumberConditioners.
@dataclass(frozen=True)
class SpecConditioning:
    cond_dim: int = 768
    t5_model_name: str = "t5-base"
    max_length: int = 128
    seconds_min: float = 0
    seconds_max: float = 512
    cross_attn_cond_ids: list[str] = field(
        default_factory=["prompt", "seconds_start", "seconds_total"].copy
    )
    global_cond_ids: list[str] = field(default_factory=["seconds_start", "seconds_total"].copy)


# Upstream inference defaults: DPM-Solver++, 250 steps, CFG 6.0.
@dataclass(frozen=True)
class SpecSampling:
    steps: int = 250
    cfg_scale: float = 6.0
    sampler_type: str = "dpmpp-3m-sde"
    sigma_min: float = 0.3
    sigma_max: float = 500.0
    rho: float = 1.0
    apg_scale: float = 0.0
    batch_cfg: bool = True
    rescale_cfg: bool = True


# Build and run params for one Stable Audio model.
@dataclass(frozen=True)
class SpecModel:
    sample_rate: int = 44100
    audio_channels: int = 2
    latent_dim: int = 64
    downsampling_ratio: int = 2048
    sample_size: int = 2097152
    # the native window; the t11 crop happens at latent staging
    generation_frames: int = 2097152 // 2048
    vae: VaeConfig = field(default_factory=VaeConfig)
    dit: SpecDit = field(default_factory=SpecDit)
    conditioning: SpecConditioning = field(default_factory=SpecConditioning)
    sampling: SpecSampling = field(default_factory=SpecSampling)
    ckpt: str = "stable-audio-open/model.safetensors"


# Decoder-only reconstruction and adversarial loss values.
@dataclass(frozen=True)
class SpecVaeLoss:
    stft_reconstruction_ffts: tuple[int, ...] = (2048, 1024, 512, 256, 128, 64, 32)
    stft_reconstruction_hops: tuple[int, ...] = (512, 256, 128, 64, 32, 16, 8)
    stft_reconstruction_windows: tuple[int, ...] = (2048, 1024, 512, 256, 128, 64, 32)
    perceptual_weighting: bool = True
    stft_reconstruction_weight: float = 1.0
    waveform_weight: float = 0.0
    adversarial_weight: float = 0.1
    feature_weight: float = 5.0
    kl_weight: float = 0.0
    discriminator_filters: int = 64
    discriminator_ffts: tuple[int, ...] = (2048, 1024, 512, 256, 128)
    discriminator_hops: tuple[int, ...] = (512, 256, 128, 64, 32)
    discriminator_windows: tuple[int, ...] = (2048, 1024, 512, 256, 128)
    # SAO config omits stride; the Stability copy defaults (1, 1)
    discriminator_stride: tuple[int, int] = (1, 1)
    discriminator_scales: int = 5
    feature_layers: int = 5


# Decoder and discriminator optimization values.
@dataclass(frozen=True)
class SpecVaeOptim:
    decoder_lr: float = 1.5e-4
    discriminator_lr: float = 3e-4
    betas: tuple[float, float] = (0.8, 0.99)
    weight_decay: float = 1e-3
    warmup_steps: int = 2000
    total_steps: int = 5000
    final_lr: float = 0.0


# Training precision and backend controls.
@dataclass(frozen=True)
class SpecVaeRuntime:
    precision: str = "bf16-mixed"
    data_workers: int = 8
    data_prefetch: int = 4
    deterministic_algorithms: bool = False
    deterministic_warn_only: bool = False
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = False
    allow_tf32: bool = False
    allow_fp16_reduced_precision_reduction: bool = False
    allow_bf16_reduced_precision_reduction: bool = True
    blas_library: str = "default"
    torch_compile: bool = False


# Decoder exponential moving average values.
@dataclass(frozen=True)
class SpecVaeEma:
    mode: str = "off"
    paper_status: str = "not_mentioned"
    code_evidence: str = "release_adjacent_wrapper_default"
    local_update_order: str = "after_optimizer_step"
    historical_run_bound: bool = False
    beta: float = 0.9999
    inverse_gamma: float = 1.0
    power: float = 0.75
    minimum: float = 0.0
    update_every: int = 1
    update_after_step: int = 1


# Published batch values and local run controls.
@dataclass(frozen=True)
class SpecVaeFit:
    sample_size: int = 65536
    chunk_seconds: float = 1.5
    batch_clips: int = 8
    # micro-batches summed into one optimizer update
    accumulation: int = 1
    crop_frames: int = 32
    steps: int = 5000
    checkpoint_every: int = 500
    logging_every: int = 10
    eval_every: int = 500
    levels: tuple[str, ...] = ("ref",)
    init_seed_index: int = -1


# Source pins and bounded artifact findings.
@dataclass(frozen=True)
class SpecVaeSource:
    paper_url: str = "https://arxiv.org/html/2407.14358v2#S4.SS1"
    config_revision: str = "f21265c1e2710b3bd2386596943f0007f55f802e"
    config_blob: str = "81ccbfd5593f990f8d64d64e361306d0d6bb07eb"
    model_sha256: str = "7b20458a071231aaf32613b6fbc7945f28f34dbba4f295bb49bad56f5f66b57e"
    release_code_commit: str = "bd6084db7b1b9895fd03b2cd554bf0c3d30ae09b"
    encodec_commit: str = "f1479a65a75c0e49e7e5d85bb1418fd57e6a9d62"
    current_vendor_commit: str = "3241adba4fc2a85cf5b29d9eb68d42f40a28e820"
    discriminator_name: str = "Encodec MultiScaleSTFTDiscriminator"
    discriminator_weight_status: str = "not found in official release artifacts"
    discriminator_weight_search_date: str = "2026-07-28"
    release_default_precision: str = "16-mixed"
    implementation_precision: str = "bf16-mixed"
    decoder_initialization: str = "pretrained_checkpoint"
    discriminator_initialization: str = "seeded_random"
    historical_run_bound: bool = False


# Held-out split and sqlite metric capture values.
@dataclass(frozen=True)
class SpecTrainTracking:
    test_clips_per_domain: int = 256
    split_seed: int = 0
    db_name: str = "train_tracking.sqlite"
    eval_batch_clips: int = 8
    metrics: tuple[str, ...] = ("fd_pann", "fad", "kl_softmax", "is_mean")
    fad_pairs: tuple[tuple[str, str], ...] = (("ref", "gen"),)


# Decoder-only adversarial training recipe.
@dataclass(frozen=True)
class SpecVaeTraining:
    model_name: str = "stable-audio-open"
    loss: SpecVaeLoss = field(default_factory=SpecVaeLoss)
    optim: SpecVaeOptim = field(default_factory=SpecVaeOptim)
    runtime: SpecVaeRuntime = field(default_factory=SpecVaeRuntime)
    ema: SpecVaeEma = field(default_factory=SpecVaeEma)
    fit: SpecVaeFit = field(default_factory=SpecVaeFit)
    source: SpecVaeSource = field(default_factory=SpecVaeSource)


# Finetune CLI values; defaults follow the generic VAE recipe.
@dataclass(frozen=True)
class TrainFinetuneConfig:
    cache_dirs: str
    data_root: Path
    out_dir: Path
    probe_remove_dc: bool = False
    probe_pad_samples: int = 0
    level: str = SpecVaeFit.levels[0]
    max_train_steps: int = SpecVaeFit.steps
    per_device_train_batch_size: int = SpecVaeFit.batch_clips
    gradient_accumulation_steps: int = SpecVaeFit.accumulation
    learning_rate: float = SpecVaeOptim.decoder_lr
    discriminator_learning_rate: float = SpecVaeOptim.discriminator_lr
    crop_frames: int = SpecVaeFit.crop_frames
    checkpointing_steps: int = SpecVaeFit.checkpoint_every
    logging_steps: int = SpecVaeFit.logging_every
    eval_steps: int = SpecVaeFit.eval_every
    data_workers: int = SpecVaeRuntime.data_workers
    data_prefetch: int = SpecVaeRuntime.data_prefetch
    channel_augment: bool = False
    decoder_start: int = 0
    seed: int = 42
    device: str = "cuda"


MODELS: dict[str, SpecModel] = {
    "stable-audio-open": SpecModel(),
}


# Typed build and sampling spec for a model name.
def spec(name: str):
    model_spec = MODELS[name]
    return model_spec


# One decoder recipe, shared by every Oobleck codec.
def get_vae_training(name: str = "stable-audio-open"):
    training = SpecVaeTraining(model_name=name)
    return training


# Split and metric capture values for decoder training.
def get_train_tracking():
    tracking = SpecTrainTracking()
    return tracking


# Native sample rate of the named model.
def get_native_sample_rate(name: str):
    sample_rate = MODELS[name].sample_rate
    return sample_rate


# Absolute checkpoint path under the data root.
def get_ckpt_path(name: str):
    rel = MODELS[name].ckpt
    if rel:
        data_root = get_data_root()
        ckpt_path = str(data_root / rel)
    else:
        ckpt_path = None
    return ckpt_path


# Stored real-music parity clip under the data root.
def get_parity_clip_path():
    data_root = get_data_root()
    path = str(data_root / "parity/music_clip.wav")
    return path


# Stored parity text prompt under the data root.
def get_parity_prompt_path():
    data_root = get_data_root()
    path = str(data_root / "parity/music_prompt.txt")
    return path
