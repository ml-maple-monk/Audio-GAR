# vendored: Stability-AI/stable-audio-tools 3241adba4fc2a85cf5b29d9eb68d42f40a28e820 (MIT)
# =============================================================================
# Vendored: Stable Audio Open — "Oobleck" VAE audio codec (tokenizer arch).
#
# Copied verbatim from the Stable Audio Open open-source implementation:
#   repo:    https://github.com/Stability-AI/stable-audio-tools
#   commit:  3241adba4fc2a85cf5b29d9eb68d42f40a28e820
#   files:   stable_audio_tools/models/blocks.py
#            stable_audio_tools/models/autoencoders.py
#            stable_audio_tools/models/bottleneck.py
# 
# =============================================================================
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils import weight_norm
from typing import Literal

from alias_free_torch import Activation1d


# -----------------------------------------------------------------------------
# Primitives  (blocks.py)
# -----------------------------------------------------------------------------
def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 0.000000001)) * (torch.sin(x * alpha) ** 2)

# Adapted from https://github.com/NVIDIA/BigVGAN/blob/main/activations.py under MIT license
# License available in LICENSES/LICENSE_NVIDIA.txt
class SnakeBeta(nn.Module):

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else: # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = snake_beta(x, alpha, beta)

        return x


def get_activation(activation: Literal["elu", "snake", "none"], antialias=False, channels=None) -> nn.Module:
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        act = SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")

    if antialias:
        act = Activation1d(act)

    return act


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, use_snake=False, antialias_activation=False, depthwise = False, bias = True):
        super().__init__()

        self.dilation = dilation

        padding = (dilation * (7-1)) // 2

        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=7, dilation=dilation, padding=padding, groups=1 if not depthwise else out_channels, bias = bias),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1, bias = bias)
        )

    def forward(self, x):
        res = x
        x = self.layers(x)
        return x + res


# -----------------------------------------------------------------------------
# Encoder / Decoder blocks  (autoencoders.py)
# -----------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False):
        super().__init__()
        self.stride = stride

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=9, use_snake=use_snake),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2)),
        )

    def forward(self, x):
        return self.layers(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False, use_nearest_upsample=False):
        super().__init__()

        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                WNConv1d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=2*stride,
                        stride=1,
                        bias=False,
                        padding='same')
            )
        else:
            upsample_layer = WNConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2))

        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            upsample_layer,
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=9, use_snake=use_snake),
        )

    def forward(self, x):
        return self.layers(x)


# -----------------------------------------------------------------------------
# Oobleck encoder / decoder  (autoencoders.py)
# -----------------------------------------------------------------------------
class OobleckEncoder(nn.Module):
    def __init__(self,
                 in_channels=2,
                 channels=128,
                 latent_dim=32,
                 c_mults = [1, 2, 4, 8],
                 strides = [2, 4, 8, 8],
                 use_snake=False,
                 antialias_activation=False
        ):
        super().__init__()
        self.in_channels = in_channels

        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=3)
        ]

        for i in range(self.depth-1):
            layers += [EncoderBlock(in_channels=c_mults[i]*channels, out_channels=c_mults[i+1]*channels, stride=strides[i], use_snake=use_snake)]

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[-1] * channels),
            WNConv1d(in_channels=c_mults[-1]*channels, out_channels=latent_dim, kernel_size=3, padding=1)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(self,
                 out_channels=2,
                 channels=128,
                 latent_dim=32,
                 c_mults = [1, 2, 4, 8],
                 strides = [2, 4, 8, 8],
                 use_snake=False,
                 antialias_activation=False,
                 use_nearest_upsample=False,
                 final_tanh=True):
        super().__init__()
        self.out_channels = out_channels

        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=latent_dim, out_channels=c_mults[-1]*channels, kernel_size=7, padding=3),
        ]

        for i in range(self.depth-1, 0, -1):
            layers += [DecoderBlock(
                in_channels=c_mults[i]*channels,
                out_channels=c_mults[i-1]*channels,
                stride=strides[i-1],
                use_snake=use_snake,
                antialias_activation=antialias_activation,
                use_nearest_upsample=use_nearest_upsample
                )
            ]

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[0] * channels),
            WNConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=3, bias=False),
            nn.Tanh() if final_tanh else nn.Identity()
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# -----------------------------------------------------------------------------
# VAE bottleneck  (bottleneck.py)
# -----------------------------------------------------------------------------
class Bottleneck(nn.Module):
    def __init__(self, is_discrete: bool = False):
        super().__init__()

        self.is_discrete = is_discrete

    def encode(self, x, return_info=False, **kwargs):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError


def vae_sample(mean, scale):
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * stdev + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return latents, kl

class VAEBottleneck(Bottleneck):
    def __init__(self):
        super().__init__(is_discrete=False)

    def encode(self, x, return_info=False, **kwargs):
        info = {}

        mean, scale = x.chunk(2, dim=1)

        x, kl = vae_sample(mean, scale)

        info["kl"] = kl

        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return x


# -----------------------------------------------------------------------------
# Autoencoder container  (autoencoders.py)
# -----------------------------------------------------------------------------
class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim,
        downsampling_ratio,
        sample_rate,
        io_channels=2,
        bottleneck: Bottleneck = None,
        pretransform: Pretransform = None,
        in_channels = None,
        out_channels = None,
        soft_clip = False,
        freeze_pretransform = False
    ):
        super().__init__()
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate

        self.latent_dim = latent_dim
        self.io_channels = io_channels
        self.in_channels = io_channels
        self.out_channels = io_channels

        self.min_length = self.downsampling_ratio

        if in_channels is not None:
            self.in_channels = in_channels

        if out_channels is not None:
            self.out_channels = out_channels

        self.bottleneck = bottleneck

        self.encoder = encoder

        self.decoder = decoder

        self.pretransform = pretransform

        self.freeze_pretransform = freeze_pretransform
        if self.pretransform is not None:
            if self.freeze_pretransform:
                for p in self.pretransform.parameters():
                    p.requires_grad = False
            else:
                for p in self.pretransform.parameters():
                    p.requires_grad = True

        self.soft_clip = soft_clip

        self.is_discrete = self.bottleneck is not None and self.bottleneck.is_discrete

    def encode(self, audio, return_info=False, skip_pretransform=False, iterate_batch=False, return_pretransform = False, **kwargs):

        info = {}
        if self.pretransform is not None and not skip_pretransform:
            if self.pretransform.enable_grad:
                if iterate_batch:
                    audios = []
                    for i in range(audio.shape[0]):
                        audios.append(self.pretransform.encode(audio[i:i+1]))
                    audio = torch.cat(audios, dim=0)
                else:
                    audio = self.pretransform.encode(audio)
            else:
                with torch.no_grad():
                    if iterate_batch:
                        audios = []
                        for i in range(audio.shape[0]):
                            audios.append(self.pretransform.encode(audio[i:i+1]))
                        audio = torch.cat(audios, dim=0)
                    else:
                        audio = self.pretransform.encode(audio)
        if self.encoder is not None:
            if iterate_batch:
                latents = []
                for i in range(audio.shape[0]):
                    latents.append(self.encoder(audio[i:i+1], **kwargs))
                latents = torch.cat(latents, dim=0)
            else:
                latents = self.encoder(audio, **kwargs)
        else:
            latents = audio
        if self.bottleneck is not None:
            # TODO: Add iterate batch logic, needs to merge the info dicts
            latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True, **kwargs)

            info.update(bottleneck_info)

        if return_info and return_pretransform:
            return latents, info, audio
        elif return_info:
            return latents, info
        elif return_pretransform:
            return latents, audio
        else:
            return latents

    def decode(self, latents, iterate_batch=False, return_loss = False, **kwargs):
        if self.bottleneck is not None:
            if iterate_batch:
                decoded = []
                for i in range(latents.shape[0]):
                    decoded.append(self.bottleneck.decode(latents[i:i+1]))
                latents = torch.cat(decoded, dim=0)
            else:
                latents = self.bottleneck.decode(latents)
        if iterate_batch:
            decoded = []
            for i in range(latents.shape[0]):
                decoded.append(self.decoder(latents[i:i+1], **kwargs))
            decoded = torch.cat(decoded, dim=0)
        else:
            if return_loss:
                decoded, loss = self.decoder(latents, **kwargs)
            else:
                decoded = self.decoder(latents, **kwargs)
        if self.pretransform is not None:
            if self.pretransform.enable_grad:
                if iterate_batch:
                    decodeds = []
                    for i in range(decoded.shape[0]):
                        decodeds.append(self.pretransform.decode(decoded[i:i+1]))
                    decoded = torch.cat(decodeds, dim=0)
                else:
                    decoded = self.pretransform.decode(decoded)
            else:
                with torch.no_grad():
                    if iterate_batch:
                        decodeds = []
                        for i in range(latents.shape[0]):
                            decodeds.append(self.pretransform.decode(decoded[i:i+1]))
                        decoded = torch.cat(decodeds, dim=0)
                    else:
                        decoded = self.pretransform.decode(decoded)

        if self.soft_clip:
            decoded = torch.tanh(decoded)
        if return_loss:
            return decoded, loss
        else:
            return decoded

    def encode_audio(self, audio, chunked=False, overlap=32, chunk_size=128, **kwargs):
        '''
        Encode audios into latents. Audios should already be preprocesed by preprocess_audio_for_encoder.
        If chunked is True, split the audio into chunks of a given maximum size chunk_size, with given overlap.
        Overlap and chunk_size params are both measured in number of latents (not audio samples)
        # and therefore you likely could use the same values with decode_audio.
        A overlap of zero will cause discontinuity artefacts. Overlap should be => receptive field size.
        Every autoencoder will have a different receptive field size, and thus ideal overlap.
        You can determine it empirically by diffing unchunked vs chunked output and looking at maximum diff.
        The final chunk may have a longer overlap in order to keep chunk_size consistent for all chunks.
        Smaller chunk_size uses less memory, but more compute.
        The chunk_size vs memory tradeoff isn't linear, and possibly depends on the GPU and CUDA version
        For example, on a A6000 chunk_size 128 is overall faster than 256 and 512 even though it has more chunks
        '''
        samples_per_latent = int(self.downsampling_ratio)
        if not chunked or audio.shape[-1] < chunk_size * samples_per_latent:
            return self.encode(audio, **kwargs)

        # chunk_size/overlap are in latent units; scale to samples for slicing.
        chunk_size_samples = chunk_size * samples_per_latent
        hop_samples = (chunk_size - overlap) * samples_per_latent
        total_samples = audio.shape[-1]

        # Anchor the final chunk to the signal end (may overlap more than `overlap`).
        chunk_starts = list(range(0, total_samples - chunk_size_samples + 1, hop_samples))
        if chunk_starts[-1] != total_samples - chunk_size_samples:
            chunk_starts.append(total_samples - chunk_size_samples)

        encoded_chunks = [self.encode(audio[..., s:s + chunk_size_samples]) for s in chunk_starts]

        total_latents = total_samples // samples_per_latent
        half_overlap_latents = overlap // 2
        output = audio.new_zeros(*encoded_chunks[0].shape[:-1], total_latents)
        num_chunks = len(chunk_starts)

        # Trim half the overlap off inner edges (edges facing a neighbour chunk).
        for i, (start_sample, chunk) in enumerate(zip(chunk_starts, encoded_chunks)):
            is_first = i == 0
            is_last = i == num_chunks - 1
            out_start = (total_latents - chunk_size) if is_last else (start_sample // samples_per_latent)
            left  = 0 if is_first else half_overlap_latents
            right = chunk_size if is_last else chunk_size - half_overlap_latents
            output[..., out_start + left : out_start + right] = chunk[..., left:right]

        return output

    def decode_audio(self, latents, chunked=False, overlap=32, chunk_size=128, **kwargs):
        '''
        Decode latents to audio.
        If chunked is True, split the latents into chunks of a given maximum size chunk_size, with given overlap, both of which are measured in number of latents.
        A overlap of zero will cause discontinuity artefacts. Overlap should be => receptive field size.
        Every autoencoder will have a different receptive field size, and thus ideal overlap.
        You can determine it empirically by diffing unchunked vs chunked audio and looking at maximum diff.
        The final chunk may have a longer overlap in order to keep chunk_size consistent for all chunks.
        Smaller chunk_size uses less memory, but more compute.
        The chunk_size vs memory tradeoff isn't linear, and possibly depends on the GPU and CUDA version
        For example, on a A6000 chunk_size 128 is overall faster than 256 and 512 even though it has more chunks
        '''
        if not chunked or latents.shape[-1] < chunk_size:
            return self.decode(latents, **kwargs)

        # chunk_size/overlap are in latent units — same as `latents`, no scaling.
        samples_per_latent = int(self.downsampling_ratio)
        hop_latents = chunk_size - overlap
        total_latents = latents.shape[-1]

        # Anchor the final chunk to the signal end (may overlap more than `overlap`).
        chunk_starts = list(range(0, total_latents - chunk_size + 1, hop_latents))
        if chunk_starts[-1] != total_latents - chunk_size:
            chunk_starts.append(total_latents - chunk_size)

        decoded_chunks = [self.decode(latents[..., s:s + chunk_size]) for s in chunk_starts]

        total_samples = total_latents * samples_per_latent
        chunk_size_samples = chunk_size * samples_per_latent
        half_overlap_samples = (overlap // 2) * samples_per_latent
        output = latents.new_zeros(*decoded_chunks[0].shape[:-1], total_samples)
        num_chunks = len(chunk_starts)

        # Trim half the overlap off inner edges (edges facing a neighbour chunk).
        for i, (start_latent, chunk) in enumerate(zip(chunk_starts, decoded_chunks)):
            is_first = i == 0
            is_last = i == num_chunks - 1
            out_start = (total_samples - chunk_size_samples) if is_last else (start_latent * samples_per_latent)
            left  = 0 if is_first else half_overlap_samples
            right = chunk_size_samples if is_last else chunk_size_samples - half_overlap_samples
            output[..., out_start + left : out_start + right] = chunk[..., left:right]

        return output


# -----------------------------------------------------------------------------
# Pretransform seam  (pretransforms.py)
# Wraps the AudioAutoencoder above as a latent pretransform for the diffusion
# model (the generator imports AutoencoderPretransform from here).
# -----------------------------------------------------------------------------
class Pretransform(nn.Module):
    def __init__(self, enable_grad, io_channels, is_discrete):
        super().__init__()

        self.is_discrete = is_discrete
        self.io_channels = io_channels
        self.encoded_channels = None
        self.downsampling_ratio = None

        self.enable_grad = enable_grad

    def forward(self, x):
        return self.encode(x)

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def tokenize(self, x):
        raise NotImplementedError

    def decode_tokens(self, tokens):
        raise NotImplementedError

class AutoencoderPretransform(Pretransform):
    def __init__(self, model, scale=1.0, model_half=False, iterate_batch=False, chunked=False, enable_grad = False):
        super().__init__(enable_grad=enable_grad, io_channels=model.io_channels, is_discrete=model.bottleneck is not None and model.bottleneck.is_discrete)
        self.model = model
        if not enable_grad:
            self.model.requires_grad_(False).eval()
        self.scale=scale
        self.downsampling_ratio = model.downsampling_ratio
        self.io_channels = model.io_channels
        self.sample_rate = model.sample_rate

        self.model_half = model_half
        self.iterate_batch = iterate_batch

        self.encoded_channels = model.latent_dim

        self.chunked = chunked
        self.num_quantizers = model.bottleneck.num_quantizers if model.bottleneck is not None and model.bottleneck.is_discrete else None
        self.codebook_size = model.bottleneck.codebook_size if model.bottleneck is not None and model.bottleneck.is_discrete else None

        if self.model_half:
            self.model.half()

    def encode(self, x, **kwargs):
        if self.model_half:
            x = x.half()
            self.model.to(torch.float16)

        encoded = self.model.encode_audio(x, chunked=self.chunked, iterate_batch=self.iterate_batch, **kwargs)

        if self.model_half:
            encoded = encoded.float()

        return encoded / self.scale

    def decode(self, z, **kwargs):
        z = z * self.scale

        if self.model_half:
            z = z.half()
            self.model.to(torch.float16)

        decoded = self.model.decode_audio(z, chunked=self.chunked, iterate_batch=self.iterate_batch, **kwargs)

        if self.model_half:
            decoded = decoded.float()

        return decoded

    def tokenize(self, x, **kwargs):
        assert self.model.is_discrete, "Cannot tokenize with a continuous model"

        _, info = self.model.encode(x, return_info = True, **kwargs)

        return info[self.model.bottleneck.tokens_id]

    def decode_tokens(self, tokens, **kwargs):
        assert self.model.is_discrete, "Cannot decode tokens with a continuous model"

        return self.model.decode_tokens(tokens, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict=strict)
