from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import click
import soundfile as sf
import torch
from safetensors import safe_open
from torch import nn

from ...runtime.dataroot import apply_data_root
from . import registry
from .vendor import tokenizer_arch
from .vendor.tokenizer_arch import (
    AudioAutoencoder,
    AutoencoderPretransform,
    OobleckDecoder,
    OobleckEncoder,
    VAEBottleneck,
)


# Stable Audio Open codec.
class StableAudioTokenizer(nn.Module):
    def __init__(self, name: str = "stable-audio-open", ckpt_path: str = "", device: str = "cuda",
                 load_pretrained: bool = True, determinism: bool = False):
        super().__init__()
        self.name = name

        self.spec = registry.spec(name)

        # same build as StableAudioGenerator
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
        self.pretransform = AutoencoderPretransform(
            autoencoder,
            scale=self.spec.vae.scale,
            iterate_batch=self.spec.vae.iterate_batch,
        )

        self.weights_path = ""
        ckpt = ckpt_path or registry.get_ckpt_path(name)
        # skipping the load leaves the constructor init
        if ckpt and load_pretrained:
            self.load_pretransform(ckpt)
            self.weights_path = ckpt

        self.pretransform.to(device)
        self.pretransform.eval()

        param_iter = self.pretransform.parameters()
        first_param = next(param_iter)
        self.device = first_param.device
        self.dtype = first_param.dtype
        if determinism:
            self.apply_determinism()
        else:
            self.apply_precision_policy()

    # Tf32 and autotuning, which determinism forbids.
    def apply_precision_policy(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Load the VAE slice of the full checkpoint only.
    def load_pretransform(self, ckpt: str):
        prefix = "pretransform.model."
        prefix_len = len(prefix)
        vae_state = {}
        with safe_open(ckpt, framework="pt", device="cpu") as source:
            for key_item in source.keys():
                if key_item.startswith(prefix):
                    vae_key = key_item[prefix_len:]
                    vae_state[vae_key] = source.get_tensor(key_item)
        self.pretransform.load_state_dict(vae_state, strict=True)

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

    # Codec sample rate in Hz.
    @property
    def sample_rate(self):
        rate = self.spec.sample_rate
        return rate

    # Latent frames per second.
    @property
    def frame_rate(self):
        rate = self.spec.sample_rate / self.spec.downsampling_ratio
        return rate

    # Waveform channel count.
    @property
    def audio_channels(self):
        channels = self.spec.audio_channels
        return channels

    # Waveform to posterior distribution.
    @torch.no_grad()
    def encode(self, wav: torch.Tensor):
        wav = wav.to(device=self.device, dtype=self.dtype)
        distribution = self.pretransform.model.encoder(wav)  # (B, 2D, F)
        return distribution

    # Vendored posterior draw: mean + eps * stdev.
    def vae_sample(self, distribution: torch.Tensor):
        # no eval-mode mean path exists upstream
        mean, scale = distribution.chunk(2, dim=1)
        latent, _ = tokenizer_arch.vae_sample(mean, scale)  # (B, D, F)
        return latent

    # Divide or multiply by the codec latent scale.
    def apply_latent_scale(self, latent: torch.Tensor, direction: str):
        scale = self.spec.vae.scale
        if direction == "normalize":
            scaled = latent / scale
        elif direction == "denormalize":
            scaled = latent * scale
        else:
            raise ValueError(f"unknown rescale direction {direction!r}")
        return scaled

    # Latent to waveform through the Oobleck decoder.
    def decode(self, latent: torch.Tensor):
        latent = latent.to(device=self.device, dtype=self.dtype)
        decoded = self.pretransform.model.decoder(latent)  # (B, C, T)
        return decoded

    # Waveform to the DiT latent space, sampled and normalized.
    @torch.no_grad()
    def encode_latent(self, wav: torch.Tensor):
        distribution = self.encode(wav)
        latent = self.vae_sample(distribution)
        normalized = self.apply_latent_scale(latent, "normalize")
        return normalized

    # Waveform to the DiT latent space, posterior draw.
    @torch.no_grad()
    def encode_latent_sample(self, wav: torch.Tensor, eps: torch.Tensor):
        # the vae_sample formula with a caller-owned eps
        distribution = self.encode(wav)
        mean, scale = distribution.chunk(2, dim=1)
        softplus_scale = nn.functional.softplus(scale)
        stdev = softplus_scale + 1e-4
        eps = eps.to(device=mean.device, dtype=mean.dtype)
        latent = mean + stdev * eps  # (B, D, F)
        normalized = self.apply_latent_scale(latent, "normalize")
        return normalized

    # DiT latent back to waveform.
    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor):
        denormalized = self.apply_latent_scale(latent, "denormalize")
        decoded = self.decode(denormalized)
        return decoded

    # Encode, draw via vendored vae_sample, decode.
    @torch.no_grad()
    def reconstruct(self, wav: torch.Tensor):
        distribution = self.encode(wav)
        latent = self.vae_sample(distribution)
        decoded = self.decode(latent)
        return decoded


# Tensors and gaps from one codec parity run.
@dataclass
class StableAudioTokenizerParityResult:
    ours: torch.Tensor
    ref: torch.Tensor
    ours_latent: torch.Tensor
    ref_latent: torch.Tensor
    latent_diff: float
    max_diff: float
    passed: bool


# Parity against AutoencoderPretransform.encode and decode.
class StableAudioTokenizerParity:
    def __init__(self, tokenizer: StableAudioTokenizer, clip_path: str, seed: int = 0,
                 device: str = "cuda"):
        self.tokenizer = tokenizer
        self.clip_path = clip_path
        self.seed = seed
        self.device = device
        self.result = None

    # Build the tokenizer and locate the stored parity clip.
    @classmethod
    def make_from_options(cls, ckpt_path: str = "", device: str = "cuda", seed: int = 0):
        tokenizer = StableAudioTokenizer(
            name="stable-audio-open", ckpt_path=ckpt_path, device=device,
        )
        # stored real-music clip; provenance sits beside it
        clip_path = registry.get_parity_clip_path()
        parity = cls(tokenizer, clip_path, seed=seed, device=device)
        return parity

    # Report whether the clip layout agrees with the spec.
    def check_clip_layout(self, clip, clip_rate: int):
        model_spec = self.tokenizer.spec
        rate_ok = clip_rate == model_spec.sample_rate
        channels_ok = clip.shape[1] == model_spec.audio_channels
        layout_ok = rate_ok and channels_ok
        if not layout_ok:
            print(f"FAIL: parity clip layout {clip.shape}@{clip_rate} disagrees with the spec")
        return layout_ok

    # Crop the clip to whole latent frames, as a batch.
    def load_audio(self, clip):
        ratio = self.tokenizer.spec.downsampling_ratio
        n_frames = clip.shape[0] // ratio
        n = ratio * n_frames
        clip_t = clip.T
        clip_crop = clip_t[None, :, :n]
        audio = torch.from_numpy(clip_crop)  # (1, C, T)
        audio = audio.to(device=self.device, dtype=self.tokenizer.dtype)
        return audio

    # Largest absolute elementwise gap between two tensors.
    def get_max_diff(self, ours: torch.Tensor, ref: torch.Tensor):
        gap = ours - ref
        gap_abs = gap.abs()
        gap_max = gap_abs.max()
        max_diff = gap_max.item()
        return max_diff

    # Run both codec arms from one seed and compare.
    def run(self):
        clip, clip_rate = sf.read(self.clip_path, dtype="float32", always_2d=True)
        layout_ok = self.check_clip_layout(clip, clip_rate)
        audio = self.load_audio(clip)

        torch.manual_seed(self.seed)
        distribution = self.tokenizer.encode(audio)
        sampled = self.tokenizer.vae_sample(distribution)
        ours_latent = self.tokenizer.apply_latent_scale(sampled, "normalize")  # (1, D, F)
        denormalized = self.tokenizer.apply_latent_scale(ours_latent, "denormalize")
        ours = self.tokenizer.decode(denormalized)  # (1, C, T)

        torch.manual_seed(self.seed)
        ref_latent = AutoencoderPretransform.encode(self.tokenizer.pretransform, audio)
        ref = AutoencoderPretransform.decode(self.tokenizer.pretransform, ref_latent)

        latent_diff = self.get_max_diff(ours_latent, ref_latent)
        max_diff = self.get_max_diff(ours, ref)
        print(
            f"ours={tuple(ours.shape)} ref={tuple(ref.shape)} "
            f"latent_max_abs_diff={latent_diff:.3e} max_abs_diff={max_diff:.3e}"
        )
        latent_ok = torch.allclose(ours_latent, ref_latent, atol=1e-4, rtol=0)
        if not latent_ok:
            print(f"FAIL: LATENT MISMATCH max_abs_diff={latent_diff}")
        # final comparison: wav [1, 2, T], atol 1e-4
        wav_ok = torch.allclose(ours, ref, atol=1e-4, rtol=0)
        if not wav_ok:
            print(f"FAIL: MISMATCH max_abs_diff={max_diff}")
        passed = layout_ok and latent_ok and wav_ok
        if passed:
            print("PASS: decomposed codec matches AutoencoderPretransform.encode/decode")
        result = StableAudioTokenizerParityResult(
            ours=ours,
            ref=ref,
            ours_latent=ours_latent,
            ref_latent=ref_latent,
            latent_diff=latent_diff,
            max_diff=max_diff,
            passed=passed,
        )
        self.result = result
        return result


@click.command()
@click.option("--data_root", type=Path, required=True)
@click.option("--ckpt_path", type=str, default="")
def main(data_root, ckpt_path):
    apply_data_root(data_root)
    parity = StableAudioTokenizerParity.make_from_options(ckpt_path=ckpt_path)
    result = parity.run()
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
