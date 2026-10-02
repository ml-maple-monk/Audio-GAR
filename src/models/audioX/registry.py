from __future__ import annotations

from dataclasses import dataclass, field

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


# Optional multimodal fusion block in front of DiT cross attention.
@dataclass(frozen=True)
class SpecGate:
    enabled: bool = False
    gate_type: str = "sparse_gated"
    num_experts_per_modality: int = 64
    num_heads: int = 24
    num_fusion_layers: int = 8

    # Gate type kwargs in the upstream wrapper's shape.
    def get_type_config(self):
        type_config = {
            "num_experts_per_modality": self.num_experts_per_modality,
            "num_heads": self.num_heads,
            "num_fusion_layers": self.num_fusion_layers,
        }
        return type_config


# Video, text and audio conditioning from one released config.
@dataclass(frozen=True)
class SpecConditioning:
    cond_dim: int = 768
    video_type: str = "clip"
    audio_type: str = "audio_autoencoder"
    clip_model_name: str = "clip-vit-base-patch32"
    t5_model_name: str = "t5-base"
    max_length: int = 128
    audio_sample_rate: int = 44100
    mel_spec_type: str = "mel_features"
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mel_channels: int = 256
    mel_target_sample_rate: int = 24000
    video_fps: int = 5
    cond_seconds: int = 10
    video_channels: int = 3
    video_size: int = 224  # CLIP input height/width
    cross_attn_cond_ids: list[str] = field(
        default_factory=["video_prompt", "text_prompt", "audio_prompt"].copy
    )
    global_cond_ids: list[str] = field(default_factory=list)


# AudioX MusicCaps protocol: DPM-Solver++ 3M-SDE, 250 steps, CFG 7.0.
@dataclass(frozen=True)
class SpecSampling:
    steps: int = 250
    cfg_scale: float = 7.0
    sampler_type: str = "dpmpp-3m-sde"
    sigma_min: float = 0.3
    sigma_max: float = 500.0
    rho: float = 1.0
    rescale_cfg: bool = True
    batch_cfg: bool = True


# Everything AudioXGenerator needs to build one model.
@dataclass(frozen=True)
class SpecModel:
    sample_rate: int = 44100
    audio_channels: int = 2
    latent_dim: int = 64
    downsampling_ratio: int = 2048
    sample_size: int = 485100
    generation_frames: int = 215
    vae: VaeConfig = field(default_factory=VaeConfig)
    dit: SpecDit = field(default_factory=SpecDit)
    conditioning: SpecConditioning = field(default_factory=SpecConditioning)
    gate: SpecGate = field(default_factory=SpecGate)
    sampling: SpecSampling = field(default_factory=SpecSampling)
    ckpt: str = "audiox/model.ckpt"
    weights_repo: str = "HKUSTAudio/AudioX"
    weights_revision: str = "3d49eff6430b739ba5a28357b1a0eedd843e6711"

    # Oobleck pretransform dict; the audio_prompt conditioner reuses it.
    def vae_pretransform_config(self):
        pretransform_dict = {
            "type": "autoencoder",
            "iterate_batch": self.vae.iterate_batch,
            "config": {
                "encoder": {
                    "type": "oobleck",
                    "requires_grad": False,
                    "config": {
                        "in_channels": self.audio_channels,
                        "channels": self.vae.channels,
                        "c_mults": self.vae.c_mults,
                        "strides": self.vae.strides,
                        "latent_dim": 2 * self.latent_dim,
                        "use_snake": self.vae.use_snake,
                    },
                },
                "decoder": {
                    "type": "oobleck",
                    "config": {
                        "out_channels": self.audio_channels,
                        "channels": self.vae.channels,
                        "c_mults": self.vae.c_mults,
                        "strides": self.vae.strides,
                        "latent_dim": self.latent_dim,
                        "use_snake": self.vae.use_snake,
                        "final_tanh": self.vae.final_tanh,
                    },
                },
                "bottleneck": {"type": "vae"},
                "latent_dim": self.latent_dim,
                "downsampling_ratio": self.downsampling_ratio,
                "io_channels": self.audio_channels,
            },
        }
        return pretransform_dict

    # Released conditioning config in the upstream factory's shape.
    def get_conditioning_dict(self):
        conditioning = self.conditioning
        if conditioning.audio_type == "audio_autoencoder":
            pretransform_dict = self.vae_pretransform_config()
            audio_config = {
                "sample_rate": conditioning.audio_sample_rate,
                "pretransform_config": pretransform_dict,
            }
        elif conditioning.audio_type == "mel_spec":
            audio_config = {
                "mel_spec_type": conditioning.mel_spec_type,
                "n_fft": conditioning.n_fft,
                "hop_length": conditioning.hop_length,
                "win_length": conditioning.win_length,
                "n_mel_channels": conditioning.n_mel_channels,
                "target_sample_rate": conditioning.mel_target_sample_rate,
            }
        else:
            raise ValueError(f"unsupported AudioX audio conditioner {conditioning.audio_type!r}")
        conditioning_dict = {
            "cond_dim": conditioning.cond_dim,
            "configs": [
                {
                    "id": "video_prompt",
                    "type": conditioning.video_type,
                    "config": {"clip_model_name": conditioning.clip_model_name},
                },
                {
                    "id": "text_prompt",
                    "type": "t5",
                    "config": {
                        "t5_model_name": conditioning.t5_model_name,
                        "max_length": conditioning.max_length,
                    },
                },
                {
                    "id": "audio_prompt",
                    "type": conditioning.audio_type,
                    "config": audio_config,
                },
            ],
        }
        return conditioning_dict

    # Instantiate the released conditioner set for this spec.
    def make_conditioner(self):
        if self.gate.enabled and self.gate.gate_type == "MAF":
            # the MAF port pulls torchaudio, absent on some images
            from .vendor.maf import make_maf_conditioner

            conditioner = make_maf_conditioner(self)
        else:
            # keeps this registry importable without the DiT stack
            from .vendor.generator_arch import create_multi_conditioner_from_conditioning_config

            conditioning_dict = self.get_conditioning_dict()
            conditioner = create_multi_conditioner_from_conditioning_config(conditioning_dict)
        return conditioner


MODELS: dict[str, SpecModel] = {
    "audiox": SpecModel(),
    "audiox-maf": SpecModel(
        generation_frames=485100 // 2048,
        dit=SpecDit(global_cond_dim=768),
        conditioning=SpecConditioning(
            video_type="clip-with-sync-w-empty-feat",
            audio_type="mel_spec",
        ),
        gate=SpecGate(enabled=True, gate_type="MAF"),
        ckpt="audiox-maf/model.ckpt",
        weights_repo="HKUSTAudio/AudioX-MAF",
        weights_revision="0a6575a6fd58039281584ad1c6f9e895233e8ca7",
    ),
    # recon ablation arm: same MAF weights, tanh-capped decoder output
    "audiox-maf-tanh": SpecModel(
        generation_frames=485100 // 2048,
        dit=SpecDit(global_cond_dim=768),
        conditioning=SpecConditioning(
            video_type="clip-with-sync-w-empty-feat",
            audio_type="mel_spec",
        ),
        gate=SpecGate(enabled=True, gate_type="MAF"),
        vae=VaeConfig(final_tanh=True),
        ckpt="audiox-maf/model.ckpt",
        weights_repo="HKUSTAudio/AudioX-MAF",
        weights_revision="0a6575a6fd58039281584ad1c6f9e895233e8ca7",
    ),
}


# Typed build and sampling spec for a model name.
def spec(name: str):
    model_spec = MODELS[name]
    return model_spec


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
