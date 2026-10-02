from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import click
import torch
from torch import nn

from ...runtime.dataroot import apply_data_root, get_data_root
from . import registry
from .vendor.tokenizer_arch import AutoencoderKL, TacotronSTFT


# Fold mel bins into channels, or restore them.
def apply_latent_fold(spec, latent: torch.Tensor, direction: str):
    bins = spec.vae.mel_bins
    if direction == "fold":
        # cache stores 3D; mel bins ride channel index c*bins+f
        batch, channels, frames, _ = latent.shape
        permuted = latent.permute(0, 1, 3, 2)
        result = permuted.reshape(batch, channels * bins, frames)  # (B, C*bins, F)
    elif direction == "unfold":
        batch, folded, frames = latent.shape
        viewed = latent.view(batch, folded // bins, bins, frames)
        result = viewed.permute(0, 1, 3, 2)  # (B, C, F, bins)
    else:
        raise ValueError(f"direction must be fold or unfold, got {direction!r}")
    return result


# Folded latent to mel through the trainable decoder side.
class MelDecoderModule(nn.Module):
    def __init__(self, vae, spec):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder
        self.spec = spec

    # Unfold, project, decode one folded latent batch.
    def forward(self, latent: torch.Tensor):
        unfolded = apply_latent_fold(self.spec, latent, "unfold")
        projected = self.post_quant_conv(unfolded)
        mel = self.decoder(projected)
        return mel


# Mel autoencoder and HiFi-GAN vocoder for tango music.
class TangoMusicTokenizer(nn.Module):
    def __init__(self, name: str = "tango-music-af-ft-mc", ckpt_path: str = "", device: str = "cuda",
                 determinism: bool = False, load_pretrained: bool = True):
        super().__init__()
        self.name = name

        self.spec = registry.spec(name)
        vae_kwargs = self.spec.vae_config()
        self.vae = AutoencoderKL(**vae_kwargs)
        stft_kwargs = asdict(self.spec.stft)
        self.stft = TacotronSTFT(**stft_kwargs)
        # the facade reads pretransform.model, as both siblings expose it
        self.pretransform = SimpleNamespace(model=self.vae)

        self.weights_path = ""
        if load_pretrained:
            # call the registry only if actually loading weights
            if ckpt_path:
                ckpt = ckpt_path
            else:
                registry_ckpt = registry.get_ckpt_path(name)
                ckpt = registry_ckpt or ""
            if ckpt:
                self.weights_path = self.load_pretransform(ckpt)

        self.vae.to(device)
        self.vae.eval()
        self.stft.to(device)
        self.stft.eval()

        first_param = next(self.vae.parameters())
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

    # Load vae and stft from a bundle or direct file.
    def load_pretransform(self, ckpt: str):
        if ckpt.endswith(".json"):
            bundle = registry.load_bundle(ckpt)
            root = get_data_root()
            vae_path = str(root / bundle.vae)
            stft_path = str(root / bundle.stft)
        else:
            vae_path = ckpt
            stft_path = registry.get_stft_ckpt_path(self.name)

        state = torch.load(vae_path, map_location="cpu", weights_only=True)
        self.vae.load_state_dict(state, strict=True)
        if stft_path:
            stft_state = torch.load(stft_path, map_location="cpu", weights_only=True)
            self.stft.load_state_dict(stft_state, strict=True)
        return ckpt

    # Deterministic kernels, no tf32, no autotuning.
    def apply_determinism(self):
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Native waveform sample rate.
    @property
    def sample_rate(self):
        return self.spec.sample_rate

    # Latent frames per second of audio.
    @property
    def frame_rate(self):
        frame_rate = self.spec.sample_rate / self.spec.downsampling_ratio
        return frame_rate

    # Waveform channel count.
    @property
    def audio_channels(self):
        return self.spec.audio_channels

    # Fold mel bins into channels, or restore them.
    def apply_latent_fold(self, latent: torch.Tensor, direction: str):
        result = apply_latent_fold(self.spec, latent, direction)
        return result

    # Scale into or out of the denoiser's latent space.
    def apply_latent_scale(self, latent: torch.Tensor, direction: str):
        scale = self.spec.vae.scale
        if direction == "normalize":
            scaled = latent * scale
        elif direction == "denormalize":
            scaled = latent / scale
        else:
            raise ValueError(f"direction must be normalize or denormalize, got {direction!r}")
        return scaled

    # Folded posterior mean and logvar, shape [B, 2*latent_dim, F].
    @torch.no_grad()
    def encode(self, wav: torch.Tensor):
        mono = wav.squeeze(1)
        mel, _, _ = self.stft.mel_spectrogram(mono)
        # off-grid frame counts derange the posterior; crop to stride
        stride = self.spec.downsampling_ratio // self.spec.stft.hop_length
        n_frames = mel.shape[-1] // stride * stride
        mel = mel[..., :n_frames]
        mel_frames_first = mel.transpose(1, 2)
        mel_image = mel_frames_first.unsqueeze(1)  # (B, 1, frames, n_mel)
        hidden = self.vae.encoder(mel_image)
        moments = self.vae.quant_conv(hidden)
        # mean rows stay below logvar rows after folding
        folded = apply_latent_fold(self.spec, moments, "fold")
        return folded

    # Draw one posterior sample from encoder moments.
    def vae_sample(self, moments: torch.Tensor):
        mean, logvar = torch.chunk(moments, 2, dim=1)
        clamped = logvar.clamp(-30.0, 20.0)
        half_logvar = 0.5 * clamped
        std = torch.exp(half_logvar)
        noise = torch.randn_like(mean)
        sample = mean + std * noise
        return sample

    # Folded latent to waveform, vocoder tail cropped.
    @torch.no_grad()
    def decode(self, latent: torch.Tensor):
        unfolded = self.apply_latent_fold(latent, "unfold")
        mel = self.vae.decode(unfolded)
        mel_squeezed = mel.squeeze(1)
        mel_input = mel_squeezed.permute(0, 2, 1)
        wav = self.vae.vocoder(mel_input)  # (B, 1, S)
        n_samples = latent.shape[-1] * self.spec.downsampling_ratio
        cropped = wav[..., :n_samples]
        return cropped

    # Scaled latent of one waveform batch.
    @torch.no_grad()
    def encode_latent(self, wav: torch.Tensor):
        moments = self.encode(wav)
        sample = self.vae_sample(moments)
        latent = self.apply_latent_scale(sample, "normalize")
        return latent

    # Posterior draw with caller-supplied noise.
    @torch.no_grad()
    def encode_latent_sample(self, wav: torch.Tensor, eps: torch.Tensor):
        moments = self.encode(wav)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        if eps.shape != mean.shape:
            raise ValueError(f"eps shape {tuple(eps.shape)} != latent shape {tuple(mean.shape)}")
        eps = eps.to(mean.device)
        clamped = logvar.clamp(-30.0, 20.0)
        half_logvar = 0.5 * clamped
        std = torch.exp(half_logvar)
        sample = mean + std * eps
        latent = self.apply_latent_scale(sample, "normalize")
        return latent

    # Decode a scaled, folded latent back to audio.
    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor):
        unscaled = self.apply_latent_scale(latent, "denormalize")
        wav = self.decode(unscaled)
        return wav

    # Encode and decode one waveform batch.
    @torch.no_grad()
    def reconstruct(self, wav: torch.Tensor):
        latent = self.encode_latent(wav)
        recon = self.decode_latent(latent)
        return recon


# Folded and unfolded decode tensors, with their gap.
@dataclass
class TangoMusicTokenizerParityResult:
    latent: torch.Tensor
    ours: torch.Tensor
    ref: torch.Tensor
    max_diff: float
    passed: bool


# Folded decode versus the donor's own unfolded path.
class TangoMusicTokenizerParity:
    def __init__(self, tokenizer: TangoMusicTokenizer, device: str = "cuda"):
        self.tokenizer = tokenizer
        self.device = device
        self.result = None

    # Build the tokenizer on one device.
    @classmethod
    def make_from_options(cls, device: str = "cuda"):
        tokenizer = TangoMusicTokenizer(device=device)
        parity = cls(tokenizer, device)
        return parity

    # Decode an unfolded latent through the donor path.
    def decode_reference(self, latent: torch.Tensor, n_samples: int):
        with torch.no_grad():
            mel = self.tokenizer.vae.decode(latent)
            mel_squeezed = mel.squeeze(1)
            mel_input = mel_squeezed.permute(0, 2, 1)
            ref = self.tokenizer.vae.vocoder(mel_input)
        ref = ref[..., :n_samples]
        return ref

    # Compare folded decode against the unfolded reference.
    def run(self):
        spec = self.tokenizer.spec
        torch.manual_seed(0)
        latent = torch.randn(
            1, spec.latent_channels, spec.generation_frames, spec.vae.mel_bins, device=self.device
        )
        folded = self.tokenizer.apply_latent_fold(latent, "fold")
        ours = self.tokenizer.decode(folded)  # (1, 1, S)
        ref = self.decode_reference(latent, ours.shape[-1])
        gap = ours - ref
        abs_gap = gap.abs()
        max_gap = abs_gap.max()
        max_diff = max_gap.item()
        print(f"fold decode max_abs_diff={max_diff:.3e}")
        passed = torch.allclose(ours, ref, atol=1e-5, rtol=0)
        if passed:
            print("PASS: folded decode matches the unfolded reference path")
        else:
            print(f"FAIL: FOLD MISMATCH max_abs_diff={max_diff}")
        self.result = TangoMusicTokenizerParityResult(
            latent=latent, ours=ours, ref=ref, max_diff=max_diff, passed=passed
        )
        return self.result


@click.command()
@click.option("--data_root", type=Path, required=True)
def main(data_root):
    apply_data_root(data_root)
    parity = TangoMusicTokenizerParity.make_from_options()
    result = parity.run()
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
