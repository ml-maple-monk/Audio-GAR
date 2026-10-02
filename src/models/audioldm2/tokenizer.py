from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import click
import torch
from torch import nn

from . import registry
from ...runtime.dataroot import apply_data_root
from .vendor.tokenizer_arch import AutoencoderKL, TacotronSTFT


VAE_PREFIX = "first_stage_model."
SCALE_KEY = "scale_factor"


# Fold mel bins into channels, or restore them.
def apply_latent_fold(spec, latent: torch.Tensor, direction: str):
    bins = spec.vae.mel_bins
    if direction == "fold":
        # cache stores 3D; mel bins ride channel index c*bins+f
        batch, channels, frames, _ = latent.shape
        permuted = latent.permute(0, 1, 3, 2)  # (B, C, F, T)
        moved = permuted.reshape(batch, channels * bins, frames)
    elif direction == "unfold":
        batch, folded, frames = latent.shape
        split = latent.view(batch, folded // bins, bins, frames)  # (B, C, F, T)
        moved = split.permute(0, 1, 3, 2)
    else:
        raise ValueError(f"direction must be fold or unfold, got {direction!r}")
    return moved


# Mel autoencoder and HiFi-GAN vocoder for audioldm2.
class AudioLDM2Tokenizer(nn.Module):
    def __init__(self, name: str = "audioldm2-full-large", ckpt_path: str = "", device: str = "cuda",
                 determinism: bool = False, load_pretrained: bool = True):
        super().__init__()
        self.name = name

        self.spec = registry.spec(name)
        vae_kwargs = self.spec.vae_config()
        self.vae = AutoencoderKL(**vae_kwargs)
        stft_kwargs = asdict(self.spec.stft)
        self.stft = TacotronSTFT(**stft_kwargs)
        # the facade reads pretransform.model, as every sibling exposes it
        self.pretransform = SimpleNamespace(model=self.vae)

        self.weights_path = ""
        self.latent_scale = 1.0
        if load_pretrained:
            # call the registry only if actually loading weights
            ckpt = ckpt_path
            if not ckpt:
                registry_ckpt = registry.get_ckpt_path(name)
                ckpt = registry_ckpt or ""
            if ckpt:
                self.weights_path = self.load_pretransform(ckpt)

        self.vae.to(device)
        self.vae.eval()
        # upstream STFT.transform forces its output to cpu; keep it there
        self.stft.to("cpu")
        self.stft.eval()

        param_iter = self.vae.parameters()
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

    # Autoencoder slice of a released AudioLDM 2 checkpoint.
    def get_vae_state(self, state: dict):
        prefix_len = len(VAE_PREFIX)
        selected = {}
        for key_item, value in state.items():
            if key_item.startswith(VAE_PREFIX):
                selected[key_item[prefix_len:]] = value
        return selected

    # Latent rescale the released checkpoint trained under.
    def get_latent_scale(self, state: dict):
        scale_tensor = state[SCALE_KEY]
        scalar = scale_tensor.reshape(())
        scale = float(scalar)
        return scale

    # Load the autoencoder slice and the latent rescale.
    def load_pretransform(self, ckpt: str):
        bundle = torch.load(ckpt, map_location="cpu", weights_only=True)
        if "state_dict" in bundle:
            state = bundle["state_dict"]
        else:
            state = bundle
        vae_state = self.get_vae_state(state)
        self.vae.load_state_dict(vae_state, strict=True)
        self.latent_scale = self.get_latent_scale(state)
        return ckpt

    # Deterministic kernels, no tf32, no autotuning.
    def apply_determinism(self):
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Native audio sample rate in Hz.
    @property
    def sample_rate(self):
        return self.spec.sample_rate

    # Latent frames per second of audio.
    @property
    def frame_rate(self):
        rate = self.spec.sample_rate / self.spec.downsampling_ratio
        return rate

    # Audio channel count the codec expects.
    @property
    def audio_channels(self):
        return self.spec.audio_channels

    # Fold mel bins into channels, or restore them.
    def apply_latent_fold(self, latent: torch.Tensor, direction: str):
        moved = apply_latent_fold(self.spec, latent, direction)
        return moved

    # Scale into or out of the denoiser's latent space.
    def apply_latent_scale(self, latent: torch.Tensor, direction: str):
        scale = self.latent_scale
        if direction == "normalize":
            scaled = latent * scale
        elif direction == "denormalize":
            scaled = latent / scale
        else:
            raise ValueError(f"direction must be normalize or denormalize, got {direction!r}")
        return scaled

    # Folded moments; raw level, unlike upstream preprocessing.
    @torch.no_grad()
    def encode(self, wav: torch.Tensor):
        detached = wav.detach()
        on_cpu = detached.to("cpu")
        squeezed = on_cpu.squeeze(1)
        source = squeezed.clamp(-1.0, 1.0)  # (B, S)
        mel, _, _, _ = self.stft.mel_spectrogram(source)
        # off-grid frame counts derange the posterior; crop to stride
        stride = self.spec.downsampling_ratio // self.spec.stft.hop_length
        n_frames = mel.shape[-1] // stride * stride
        mel = mel[..., :n_frames]
        mel = mel.to(self.device)
        transposed = mel.transpose(1, 2)
        mel_image = transposed.unsqueeze(1)  # (B, 1, T, M)
        hidden = self.vae.encoder(mel_image)
        moments = self.vae.quant_conv(hidden)
        # mean rows stay below logvar rows after folding
        folded = apply_latent_fold(self.spec, moments, "fold")
        return folded

    # Draw one posterior sample from encoder moments.
    def vae_sample(self, moments: torch.Tensor):
        mean, logvar = torch.chunk(moments, 2, dim=1)
        clamped = logvar.clamp(-30.0, 20.0)
        std = torch.exp(0.5 * clamped)
        noise = torch.randn_like(mean)
        sample = mean + std * noise
        return sample

    # Folded latent to waveform, vocoder tail cropped.
    @torch.no_grad()
    def decode(self, latent: torch.Tensor):
        unfolded = self.apply_latent_fold(latent, "unfold")
        mel = self.vae.decode(unfolded)
        squeezed = mel.squeeze(1)
        mel_frames = squeezed.permute(0, 2, 1)
        wav = self.vae.vocoder(mel_frames)
        n_samples = latent.shape[-1] * self.spec.downsampling_ratio
        cropped = wav[..., :n_samples]
        return cropped

    # Scaled latent of one waveform batch.
    @torch.no_grad()
    def encode_latent(self, wav: torch.Tensor):
        moments = self.encode(wav)
        sample = self.vae_sample(moments)
        scaled = self.apply_latent_scale(sample, "normalize")
        return scaled

    # Posterior draw with caller-supplied noise.
    @torch.no_grad()
    def encode_latent_sample(self, wav: torch.Tensor, eps: torch.Tensor):
        moments = self.encode(wav)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        if eps.shape != mean.shape:
            raise ValueError(f"eps shape {tuple(eps.shape)} != latent shape {tuple(mean.shape)}")
        eps = eps.to(mean.device)
        clamped = logvar.clamp(-30.0, 20.0)
        std = torch.exp(0.5 * clamped)
        sample = mean + std * eps
        scaled = self.apply_latent_scale(sample, "normalize")
        return scaled

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


# Tensors and verdict of one fold-decode parity check.
@dataclass
class AudioLDM2TokenizerParityResult:
    latent: torch.Tensor
    folded: torch.Tensor
    ours: torch.Tensor
    ref: torch.Tensor
    max_diff: float
    passed: bool


# Fold decode versus the donor's own unfolded path.
class AudioLDM2TokenizerParity:
    def __init__(self, tokenizer: AudioLDM2Tokenizer, device: str = "cuda"):
        self.tokenizer = tokenizer
        self.device = device
        self.result = None

    # Build the default tokenizer on one device.
    @classmethod
    def make_from_options(cls, device: str = "cuda"):
        tokenizer = AudioLDM2Tokenizer(device=device)
        parity = cls(tokenizer, device=device)
        return parity

    # Decode through the vae and vocoder, no fold.
    def decode_reference(self, latent: torch.Tensor, n_samples: int):
        with torch.no_grad():
            mel = self.tokenizer.vae.decode(latent)
            squeezed = mel.squeeze(1)
            mel_frames = squeezed.permute(0, 2, 1)
            ref = self.tokenizer.vae.vocoder(mel_frames)
        cropped = ref[..., :n_samples]
        return cropped

    # Decode one seeded latent both ways and compare.
    def run(self):
        spec = self.tokenizer.spec
        torch.manual_seed(0)
        latent = torch.randn(
            1, spec.latent_channels, spec.generation_frames, spec.vae.mel_bins, device=self.device
        )
        folded = self.tokenizer.apply_latent_fold(latent, "fold")
        ours = self.tokenizer.decode(folded)
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
        result = AudioLDM2TokenizerParityResult(
            latent=latent, folded=folded, ours=ours, ref=ref, max_diff=max_diff, passed=passed,
        )
        self.result = result
        return result


# Run the fold-decode parity check on the default checkpoint.
@click.command()
@click.option("--data_root", type=Path, required=True)
def main(data_root):
    apply_data_root(data_root)
    parity = AudioLDM2TokenizerParity.make_from_options()
    result = parity.run()
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
