from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..dataset.registry import VaeBatch
from ..models.stable_audio_open.registry import SpecVaeLoss
from ..models.stable_audio_open.vendor.encodec import MultiScaleSTFTDiscriminator
from ..models.tango2.registry import SpecVocoderLoss
from .vendor.hifigan.models import discriminator_loss, feature_loss, generator_loss
from .vendor.stable_audio_tools.training.losses import auraloss


@dataclass(frozen=True)
class LossStftReconstructionModules:
    sum_difference: nn.Module
    channel: nn.Module


@dataclass(frozen=True)
class LossDiscriminator:
    discriminator: torch.Tensor
    adversarial: torch.Tensor
    feature_matching: torch.Tensor


@dataclass
class GeneratorLossResult:
    total: torch.Tensor
    value_dict: dict


@dataclass
class VocoderGenLoss:
    mel: torch.Tensor
    adversarial: torch.Tensor
    feature_matching: torch.Tensor


@dataclass(frozen=True)
class DiscriminatorOutput:
    logit_list: list
    feature_list: list


class StoredActivationDiscriminator(MultiScaleSTFTDiscriminator):
    # Score audio through every scale, keeping all activations.
    def forward(self, audio: torch.Tensor):
        logit_list = []
        feature_list = []
        for scale_item in self.discriminators:
            mapped_list = []
            value = scale_item.spec_transform(audio)  # (B, C, F, T) complex
            if scale_item.spec_scale_pow != 0.0:
                magnitude = value.abs() + 1e-6
                weight = torch.pow(magnitude, scale_item.spec_scale_pow)
                value = value * weight
            joined = torch.cat([value.real, value.imag], dim=1)  # (B, 2C, F, T)
            value = joined.transpose(-2, -1)  # (B, 2C, T, F)
            for layer_item in scale_item.convs:
                value = layer_item(value)
                value = scale_item.activation(value)
                mapped_list.append(value)
            value = scale_item.conv_post(value)  # (B, 1, T, F)
            logit_list.append(value)
            feature_list.append(mapped_list)
        output = DiscriminatorOutput(logit_list=logit_list, feature_list=feature_list)
        return output


class LossFactory:
    def __init__(self, spec: SpecVaeLoss):
        self.spec = spec

    # Pinned five-scale EnCodec discriminator.
    def make_discriminator_model(self, audio_channels: int):
        spec = self.spec
        discriminator_model = StoredActivationDiscriminator(
            in_channels=audio_channels,
            filters=spec.discriminator_filters,
            n_ffts=list(spec.discriminator_ffts),
            hop_lengths=list(spec.discriminator_hops),
            win_lengths=list(spec.discriminator_windows),
            stride=spec.discriminator_stride,
        )
        return discriminator_model

    # Pinned stereo STFT reconstruction losses.
    def make_stft_reconstruction(self, sample_rate: int):
        spec = self.spec
        kwargs = {
            "fft_sizes": list(spec.stft_reconstruction_ffts),
            "hop_sizes": list(spec.stft_reconstruction_hops),
            "win_lengths": list(spec.stft_reconstruction_windows),
            "perceptual_weighting": spec.perceptual_weighting,
            "sample_rate": sample_rate,
        }
        sum_difference = auraloss.SumAndDifferenceSTFTLoss(**kwargs)
        channel = auraloss.MultiResolutionSTFTLoss(**kwargs)
        modules = LossStftReconstructionModules(sum_difference=sum_difference, channel=channel)
        return modules


class DiscriminatorLossTerms:
    # Mean absolute gap over one scale's layer activations.
    def get_scale_feature_gap(self, ground_truth_scale: list, reconstructed_scale: list):
        gap_sum = 0
        layer_pairs = zip(ground_truth_scale, reconstructed_scale)
        for ground_truth_layer, reconstructed_layer in layer_pairs:
            difference = ground_truth_layer - reconstructed_layer  # (B, C, T, F)
            layer_gap = difference.abs().mean()  # ()
            gap_sum = gap_sum + layer_gap
        scale_gap = gap_sum / len(ground_truth_scale)
        return scale_gap

    # Current upstream EnCodec terms reduced over all scales.
    def compute_terms(self, discriminator_model: nn.Module, ground_truth_audio: torch.Tensor,
                      reconstructed_audio: torch.Tensor):
        ground_truth = discriminator_model(ground_truth_audio)
        reconstructed = discriminator_model(reconstructed_audio)
        device = ground_truth_audio.device
        feature_matching = torch.tensor(0.0, device=device)
        discriminator = torch.tensor(0.0, device=device)
        adversarial = torch.tensor(0.0, device=device)
        scale_pairs = zip(ground_truth.feature_list, reconstructed.feature_list)
        for scale_idx, (ground_truth_scale, reconstructed_scale) in enumerate(scale_pairs):
            scale_gap = self.get_scale_feature_gap(ground_truth_scale, reconstructed_scale)
            feature_matching = feature_matching + scale_gap
            real_logit = ground_truth.logit_list[scale_idx]  # (B, 1, T, F)
            fake_logit = reconstructed.logit_list[scale_idx]  # (B, 1, T, F)
            real_hinge = F.relu(1 - real_logit)  # (B, 1, T, F)
            fake_hinge = F.relu(1 + fake_logit)  # (B, 1, T, F)
            discriminator = discriminator + real_hinge.mean()
            discriminator = discriminator + fake_hinge.mean()
            adversarial = adversarial - fake_logit.mean()
        num_scales = len(ground_truth.logit_list)
        terms = LossDiscriminator(
            discriminator=discriminator / num_scales,
            adversarial=adversarial / num_scales,
            feature_matching=feature_matching / num_scales,
        )
        return terms


class MelCodecView:
    def __init__(self, codec):
        self.codec = codec

    # Family mel in the encoder's image orientation.
    def make_mel(self, wav: torch.Tensor):
        squeezed = wav.squeeze(1)  # (B, S)
        mel_result = self.codec.stft.mel_spectrogram(squeezed)
        mel = mel_result[0]  # (B, M, T)
        transposed = mel.transpose(1, 2)  # (B, T, M)
        image = transposed.unsqueeze(1)  # (B, 1, T, M)
        return image

    # Ground-truth mel, gradient free.
    def make_target(self, wav: torch.Tensor):
        with torch.no_grad():
            image = self.make_mel(wav)  # (B, 1, T, M)
        return image


class MelGradTransform:
    def __init__(self, codec):
        self.codec = codec

    # Same conv path as STFT.transform, epsilon-stable magnitude.
    def make_magnitude(self, stft_fn, wav: torch.Tensor, eps: float):
        pad = stft_fn.filter_length // 2
        unsqueezed = wav.unsqueeze(1)  # (B, 1, S)
        padded = F.pad(unsqueezed, (pad, pad), mode="reflect")  # (B, 1, S + 2P)
        transformed = F.conv1d(padded, stft_fn.forward_basis, stride=stft_fn.hop_length)  # (B, 2K, T)
        cutoff = stft_fn.filter_length // 2 + 1
        real_part = transformed[:, :cutoff, :]  # (B, K, T)
        imag_part = transformed[:, cutoff:, :]  # (B, K, T)
        power = real_part**2 + imag_part**2 + eps  # (B, K, T)
        magnitude = torch.sqrt(power)  # (B, K, T)
        return magnitude

    # Family mel that keeps the gradient to the waveform.
    def make_mel(self, wav: torch.Tensor):
        stft = self.codec.stft
        squeezed = wav.squeeze(1)  # (B, S)
        magnitude = self.make_magnitude(stft.stft_fn, squeezed, 1e-9)  # (B, K, T)
        projected = torch.matmul(stft.mel_basis, magnitude)  # (B, M, T)
        mel = stft.spectral_normalize(projected, torch.log)  # (B, M, T)
        transposed = mel.transpose(1, 2)  # (B, T, M)
        image = transposed.unsqueeze(1)  # (B, 1, T, M)
        return image


class VocoderLossSet:
    def __init__(self, modules: nn.Module, mel_view: MelCodecView,
                 mel_grad: MelGradTransform, loss_spec: SpecVocoderLoss):
        self.modules = modules
        self.mel_view = mel_view
        self.mel_grad = mel_grad
        self.loss_spec = loss_spec

    # Hinge loss of both waveform critics on one pair.
    def make_critic_loss(self, wav_true: torch.Tensor, wav_pred: torch.Tensor):
        mpd_out = self.modules.mpd_model(wav_true, wav_pred)
        msd_out = self.modules.msd_model(wav_true, wav_pred)
        mpd_loss = discriminator_loss(mpd_out[0], mpd_out[1])
        msd_loss = discriminator_loss(msd_out[0], msd_out[1])
        loss = mpd_loss[0] + msd_loss[0]  # ()
        return loss

    # Mel, adversarial and feature terms, all differentiable.
    def make_generator_loss(self, wav_true: torch.Tensor, wav_pred: torch.Tensor):
        mel_of_true = self.mel_view.make_target(wav_true)  # (B, 1, T, M)
        mel_of_pred = self.mel_grad.make_mel(wav_pred)  # (B, 1, T, M)
        mel_gap = F.l1_loss(mel_of_true, mel_of_pred)  # ()
        mel = mel_gap * self.loss_spec.mel_weight  # ()
        mpd_out = self.modules.mpd_model(wav_true, wav_pred)
        msd_out = self.modules.msd_model(wav_true, wav_pred)
        mpd_feature = feature_loss(mpd_out[2], mpd_out[3])
        msd_feature = feature_loss(msd_out[2], msd_out[3])
        feature_matching = mpd_feature + msd_feature  # ()
        mpd_adversarial = generator_loss(mpd_out[1])
        msd_adversarial = generator_loss(msd_out[1])
        adversarial = mpd_adversarial[0] + msd_adversarial[0]  # ()
        result = VocoderGenLoss(mel=mel, adversarial=adversarial, feature_matching=feature_matching)
        return result


class JointGeneratorLoss:
    def __init__(self, loss_set: VocoderLossSet, autocast: bool, autocast_dtype: torch.dtype):
        self.loss_set = loss_set
        self.autocast = autocast
        self.autocast_dtype = autocast_dtype

    # All three HiFi-GAN generator terms, differentiable.
    def make_loss(self, batch: VaeBatch, wav_pred: torch.Tensor):
        kind = batch.latent.device.type
        with torch.autocast(device_type=kind, dtype=self.autocast_dtype, enabled=self.autocast):
            terms = self.loss_set.make_generator_loss(batch.ground_truth, wav_pred)
            total = terms.adversarial + terms.feature_matching + terms.mel  # ()
        value_dict = {
            "vocoder_mel": float(terms.mel.detach()),
            "vocoder_adversarial": float(terms.adversarial.detach()),
            "vocoder_feature_matching": float(terms.feature_matching.detach()),
            "vocoder_total": float(total.detach()),
        }
        result = GeneratorLossResult(total=total, value_dict=value_dict)
        return result
