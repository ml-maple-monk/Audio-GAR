from __future__ import annotations

from dataclasses import dataclass, field

from ...runtime.dataroot import get_data_root


# config targets name the vendored copy, so upstream stays unedited
VENDOR_ROOT = f"{__package__}.vendor.audioldm2"


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
    encoder_mel_bins: int = 64
    mel_bins: int = 16


# Cross-attention UNet over the two conditioning streams.
@dataclass(frozen=True)
class SpecUnet:
    image_size: int = 64
    in_channels: int = 8
    out_channels: int = 8
    model_channels: int = 128
    attention_resolutions: list[int] = field(default_factory=[8, 4, 2].copy)
    num_res_blocks: int = 2
    channel_mult: list[int] = field(default_factory=[1, 2, 3, 5].copy)
    num_head_channels: int = 32
    use_spatial_transformer: bool = True
    transformer_depth: int = 2
    # the large release adds a third, contextless cross-attention slot
    context_dim: list[int | None] = field(default_factory=[768, 1024, None].copy)
    # v1 film conditioning; v2 leaves the vendor default
    extra_film_condition_dim: int | None = None
    # v2-only stacked self-attention block; v1 has none
    extra_sa_layer: bool = True


# Noise schedule the released checkpoint was trained under.
@dataclass(frozen=True)
class SpecSchedule:
    timesteps: int = 1000
    linear_start: float = 0.0015
    linear_end: float = 0.0195
    beta_schedule: str = "linear"
    parameterization: str = "eps"
    cosine_s: float = 8e-3


# CLAP and FLAN-T5 feed a GPT-2 sequence generator.
@dataclass(frozen=True)
class SpecConditioning:
    clap_amodel: str = "HTSAT-base"
    clap_sampling_rate: int = 48000
    clap_embed_mode: str = "text"
    text_encoder: str = "audioldm2-full-large/flan-t5-large"
    sequence_model: str = "audioldm2-full-large/gpt2"
    text_tokenizer: str = "audioldm2-full-large/roberta-base"
    hub_text_encoder: str = "google/flan-t5-large"
    hub_sequence_model: str = "gpt2"
    hub_text_tokenizer: str = "roberta-base"
    sequence_gen_length: int = 8
    sequence_input_key: list[str] = field(
        default_factory=["film_clap_cond1", "crossattn_flan_t5"].copy
    )
    sequence_input_embed_dim: list[int] = field(default_factory=[512, 1024].copy)
    cond_stage_keys: list[str] = field(
        default_factory=["crossattn_audiomae_generated", "crossattn_flan_t5"].copy
    )
    # v1 conditioning: one film CLAP entry, no sequence generator
    film_only: bool = False


# Released AudioLDM 2 protocol: 200 DDIM steps, guidance 3.5.
@dataclass(frozen=True)
class SpecSampling:
    steps: int = 200
    cfg_scale: float = 3.5
    sampler_type: str = "ddim"
    ddim_eta: float = 1.0
    sigma_min: float = 0.0
    sigma_max: float = 0.0
    rho: float = 0.0
    rescale_cfg: bool = False
    batch_cfg: bool = False
    candidates: int = 3
    # which scorer ranks best-of-n candidates
    candidate_scorer: str = "clap"


# Everything the audioldm2 wrappers need to build one model.
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
    schedule: SpecSchedule = field(default_factory=SpecSchedule)
    conditioning: SpecConditioning = field(default_factory=SpecConditioning)
    sampling: SpecSampling = field(default_factory=SpecSampling)
    ckpt: str = "audioldm2-full-large/audioldm2-full-large-1150k.pth"
    weights_repo: str = "haoheliu/audioldm2-full-large-1150k"
    weights_file: str = "audioldm2-full-large-1150k.pth"
    # generation reads EMA weights, as the released eval path does
    use_ema: bool = False
    # v1 stores one cond tree; empty means no remap
    legacy_cond_prefix: str = ""
    # non-learned cond buffers this checkpoint line never stored
    cond_allowed_missing: tuple[str, ...] = ()
    # scorer clap untrained for v1; tolerate its missing weights
    allow_missing_scorer: bool = False

    # Folded channel count the latent cache stores.
    @property
    def latent_dim(self):
        dim = self.latent_channels * self.vae.mel_bins
        return dim

    # Mel frames one generation decodes to.
    @property
    def mel_frames(self):
        frames = self.generation_frames * 4
        return frames

    # Wall clock one native generation covers.
    @property
    def seconds_total(self):
        seconds = self.generation_frames * self.downsampling_ratio / self.sample_rate
        return seconds

    # Autoencoder config in the upstream constructor's shape.
    def vae_config(self):
        config = {
            "ddconfig": {
                "double_z": self.vae.double_z,
                "mel_bins": self.vae.encoder_mel_bins,
                "z_channels": self.vae.z_channels,
                "resolution": self.vae.resolution,
                "downsample_time": False,
                "in_channels": self.vae.in_channels,
                "out_ch": self.vae.out_channels,
                "ch": self.vae.channels,
                "ch_mult": list(self.vae.ch_mult),
                "num_res_blocks": self.vae.num_res_blocks,
                "attn_resolutions": [],
                "dropout": self.vae.dropout,
            },
            "embed_dim": self.vae.embed_dim,
            "image_key": "fbank",
            "subband": self.vae.subband,
            "time_shuffle": 1,
            "sampling_rate": self.sample_rate,
            "batchsize": 4,
        }
        return config

    # Denoiser config in the upstream constructor's shape.
    def unet_config(self):
        config = {
            "image_size": self.unet.image_size,
            "in_channels": self.unet.in_channels,
            "out_channels": self.unet.out_channels,
            "model_channels": self.unet.model_channels,
            "attention_resolutions": list(self.unet.attention_resolutions),
            "num_res_blocks": self.unet.num_res_blocks,
            "channel_mult": list(self.unet.channel_mult),
            "num_head_channels": self.unet.num_head_channels,
            "use_spatial_transformer": self.unet.use_spatial_transformer,
            "transformer_depth": self.unet.transformer_depth,
            "context_dim": list(self.unet.context_dim),
            "extra_sa_layer": self.unet.extra_sa_layer,
            "extra_film_condition_dim": self.unet.extra_film_condition_dim,
        }
        return config

    # Released conditioning tree, in the upstream factory's shape.
    def cond_stage_config(self):
        module = f"{VENDOR_ROOT}.latent_diffusion.modules.encoders.modules"
        if self.conditioning.film_only:
            # v1: one top-level film CLAP entry, no sequence generator
            config = {"film_clap_cond1": self.get_clap_conditioner(module)}
        else:
            config = self.get_sequence_config(module)
        return config

    # Two-stream v2 tree around the GPT-2 sequence generator.
    def get_sequence_config(self, module: str):
        cond = self.conditioning
        config = {
            "crossattn_audiomae_generated": {
                "cond_stage_key": "all",
                "conditioning_key": "crossattn",
                "target": f"{module}.SequenceGenAudioMAECond",
                "params": {
                    "always_output_audiomae_gt": False,
                    "learnable": True,
                    "use_gt_mae_output": True,
                    "use_gt_mae_prob": 0.0,
                    "base_learning_rate": 0.0002,
                    "sequence_gen_length": cond.sequence_gen_length,
                    "use_warmup": True,
                    "sequence_input_key": list(cond.sequence_input_key),
                    "sequence_input_embed_dim": list(cond.sequence_input_embed_dim),
                    "batchsize": 16,
                    "cond_stage_config": self.get_sequence_conditioners(module),
                },
            },
            "crossattn_flan_t5": self.get_text_conditioner(module),
        }
        return config

    # Film-conditioned CLAP text tower, shared by v1 and v2.
    def get_clap_conditioner(self, module: str):
        cond = self.conditioning
        conditioner = {
            "cond_stage_key": "text",
            "conditioning_key": "film",
            "target": f"{module}.CLAPAudioEmbeddingClassifierFreev2",
            "params": {
                "sampling_rate": cond.clap_sampling_rate,
                "embed_mode": cond.clap_embed_mode,
                "amodel": cond.clap_amodel,
                # upstream defaults to 0.1 and drops rows at inference too
                "unconditional_prob": 0.0,
            },
        }
        return conditioner

    # FLAN-T5 hidden states, from the staged encoder.
    def get_text_conditioner(self, module: str):
        encoder_path = get_asset_path(self, "text_encoder")
        conditioner = {
            "cond_stage_key": "text",
            "conditioning_key": "crossattn",
            "target": f"{module}.FlanT5HiddenState",
            "params": {"text_encoder_name": encoder_path},
        }
        return conditioner

    # The three inputs the GPT-2 sequence generator reads.
    def get_sequence_conditioners(self, module: str):
        audiomae = {
            "cond_stage_key": "ta_kaldi_fbank",
            "conditioning_key": "crossattn",
            "target": f"{module}.AudioMAEConditionCTPoolRand",
            "params": {
                "regularization": False,
                "no_audiomae_mask": True,
                "time_pooling_factors": [8],
                "freq_pooling_factors": [8],
                "eval_time_pooling": 8,
                "eval_freq_pooling": 8,
                "mask_ratio": 0,
            },
        }
        conditioners = {
            "film_clap_cond1": self.get_clap_conditioner(module),
            "crossattn_flan_t5": self.get_text_conditioner(module),
            "crossattn_audiomae_pooled": audiomae,
        }
        return conditioners

    # LatentDiffusion kwargs, as the release builds them.
    def get_model_params(self, device: str):
        first_stage = {
            "target": f"{VENDOR_ROOT}.latent_encoder.autoencoder.AutoencoderKL",
            "params": self.vae_config(),
        }
        unet = {
            "target": f"{VENDOR_ROOT}.latent_diffusion.modules"
                      ".diffusionmodules.openaimodel.UNetModel",
            "params": self.unet_config(),
        }
        params = {
            "first_stage_config": first_stage,
            "unet_config": unet,
            "cond_stage_config": self.cond_stage_config(),
            "device": device,
            "sampling_rate": self.sample_rate,
            "batchsize": 16,
            "base_learning_rate": 0.0001,
            "warmup_steps": 5000,
            "optimize_ddpm_parameter": True,
            "linear_start": self.schedule.linear_start,
            "linear_end": self.schedule.linear_end,
            "num_timesteps_cond": 1,
            "log_every_t": 200,
            "timesteps": self.schedule.timesteps,
            "unconditional_prob_cfg": 0.1,
            "parameterization": self.schedule.parameterization,
            "first_stage_key": "fbank",
            "latent_t_size": self.generation_frames,
            "latent_f_size": self.vae.mel_bins,
            "channels": self.latent_channels,
            "monitor": "val/loss_simple_ema",
            "scale_by_std": True,
            "evaluation_params": {
                "unconditional_guidance_scale": self.sampling.cfg_scale,
                "ddim_sampling_steps": self.sampling.steps,
                "n_candidates_per_samples": 3,
            },
        }
        return params


MODELS: dict[str, SpecModel] = {
    "audioldm2-full-large": SpecModel(),
    # v1 release; film conditioning, generation wired to the film cond
    "audioldm1-s-full": SpecModel(
        unet=SpecUnet(
            context_dim=[None], transformer_depth=1,
            extra_film_condition_dim=512, extra_sa_layer=False,
        ),
        conditioning=SpecConditioning(
            film_only=True,
            clap_amodel="HTSAT-tiny",
            # released v1 CLAP conditioner runs at 16 kHz
            clap_sampling_rate=16000,
            clap_embed_mode="text",
            text_tokenizer="audioldm2-full/roberta-base",
            hub_text_tokenizer="roberta-base",
        ),
        # conditioning CLAP scores 3 candidates for v1
        sampling=SpecSampling(steps=200, cfg_scale=2.5, candidates=3, candidate_scorer="cond"),
        # released eval swaps in EMA weights before sampling
        use_ema=True,
        legacy_cond_prefix="cond_stage_model.",
        cond_allowed_missing=(
            "mel_transform.spectrogram.window",
            "mel_transform.mel_scale.fb",
        ),
        # this checkpoint never stored the scorer's weights
        allow_missing_scorer=True,
        ckpt="audioldm1/audioldm-s-full.ckpt",
        weights_repo="haoheliu/AudioLDM-S-Full",
        weights_file="audioldm-s-full",
    ),
    # earlier snapshot of the large release
    "audioldm2-full-large-650k": SpecModel(
        ckpt="audioldm2-full-large/audioldm2-full-large-650k.pth",
        weights_repo="haoheliu/audioldm2-full-large-650k",
        weights_file="audioldm2-full-large-650k.pth",
    ),
    # base release; two conditioning slots, shallow transformer
    "audioldm2-full-base": SpecModel(
        unet=SpecUnet(context_dim=[768, 1024], transformer_depth=1),
        ckpt="audioldm2-full/audioldm2-full.pth",
        weights_repo="haoheliu/audioldm2-full",
        weights_file="audioldm2-full.pth",
    ),
    # base arch specialized on music data
    "audioldm2-music-665k": SpecModel(
        unet=SpecUnet(context_dim=[768, 1024], transformer_depth=1),
        ckpt="audioldm2-music/audioldm2-music-665k.pth",
        weights_repo="haoheliu/audioldm2-music-665k",
        weights_file="audioldm2-music-665k.pth",
    ),
}


# Staged directory for one auxiliary asset, or its hub id.
def get_asset_path(model: SpecModel, asset: str):
    relative = getattr(model.conditioning, asset)
    fallback = getattr(model.conditioning, f"hub_{asset}")
    # get_data_root raises a bare Exception when no root is set
    try:
        data_root = get_data_root()
        staged = data_root / relative
    except Exception:
        staged = None
    asset_path = fallback
    if staged is not None and staged.is_dir():
        asset_path = str(staged)
    return asset_path


# Typed build and sampling spec for a model name.
def spec(name: str):
    model_spec = MODELS[name]
    return model_spec


# Native sample rate of one named model.
def get_native_sample_rate(name: str):
    model_spec = MODELS[name]
    return model_spec.sample_rate


# Absolute checkpoint path under the injected data root.
def get_ckpt_path(name: str):
    rel = MODELS[name].ckpt
    ckpt_path = None
    if rel:
        data_root = get_data_root()
        staged = data_root / rel
        ckpt_path = str(staged)
    return ckpt_path
