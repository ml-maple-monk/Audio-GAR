# vendored: ZeyueT/AudioX 3bdfb7081636b9e62224039e37dadaa264dc781f (CC-BY-NC)
"""AudioX-MAF modules the released DiT checkpoint needs."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from torchaudio import transforms

from .generator_arch import (
    CLIPConditioner,
    Conditioner,
    MultiConditioner,
    T5Conditioner,
)


class MAFBlock(nn.Module):
    """Released modality-aware fusion block."""

    def __init__(
        self,
        dim: int,
        num_experts_per_modality: int,
        num_heads: int,
        num_fusion_layers: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts_per_modality = num_experts_per_modality
        total_experts = num_experts_per_modality * 3

        self.gating_network = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 3), nn.Sigmoid()
        )
        self.unified_experts = nn.Parameter(torch.randn(total_experts, dim))
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_fusion_layers
        )
        self.norm_v2 = nn.LayerNorm(dim)
        self.norm_t2 = nn.LayerNorm(dim)
        self.norm_a2 = nn.LayerNorm(dim)
        self.bypass_gate_v = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_t = nn.Parameter(torch.tensor(-10.0))
        self.bypass_gate_a = nn.Parameter(torch.tensor(-10.0))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.MultiheadAttention) and hasattr(module, "out_proj"):
            nn.init.zeros_(module.out_proj.weight)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)

    def forward(
        self,
        video_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size = video_tokens.shape[0]
        all_global = torch.cat(
            [
                video_tokens.mean(dim=1),
                text_tokens.mean(dim=1),
                audio_tokens.mean(dim=1),
            ],
            dim=1,
        )
        w_v, w_t, w_a = self.gating_network(all_global).chunk(3, dim=-1)
        full_context = torch.cat(
            [
                video_tokens * w_v.unsqueeze(-1),
                text_tokens * w_t.unsqueeze(-1),
                audio_tokens * w_a.unsqueeze(-1),
            ],
            dim=1,
        )
        experts = self.unified_experts.unsqueeze(0).expand(batch_size, -1, -1)
        info, _ = self.cross_attn(experts, full_context, full_context)
        fused = self.fusion_transformer(self.norm1(experts + info))
        fused_v, fused_t, fused_a = fused.chunk(3, dim=1)
        final_v = video_tokens + torch.sigmoid(self.bypass_gate_v) * self.norm_v2(
            fused_v.mean(dim=1)
        ).unsqueeze(1)
        final_t = text_tokens + torch.sigmoid(self.bypass_gate_t) * self.norm_t2(
            fused_t.mean(dim=1)
        ).unsqueeze(1)
        final_a = audio_tokens + torch.sigmoid(self.bypass_gate_a) * self.norm_a2(
            fused_a.mean(dim=1)
        ).unsqueeze(1)
        return {"video": final_v, "text": final_t, "audio": final_a}


class MAFCLIPConditioner(CLIPConditioner):
    """CLIP conditioner with the two Synchformer projection parameters."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.proj_sync = nn.Linear(in_features=240, out_features=self.out_features)
        self.sync_weight = nn.Parameter(torch.tensor(0.0))


def get_mel_spectrogram(
    waveform: torch.Tensor,
    n_fft: int = 1024,
    n_mel_channels: int = 100,
    target_sample_rate: int = 24000,
    hop_length: int = 256,
    win_length: int = 1024,
) -> torch.Tensor:
    """Released AudioX mel feature transform."""
    mel_stft = transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mel_channels,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    ).to(waveform.device)
    if waveform.ndim == 3:
        waveform = waveform.mean(dim=1)
    if waveform.ndim != 2:
        raise ValueError(f"mel prompt must be [B,C,T] or [B,T], got {tuple(waveform.shape)}")
    mel = mel_stft(waveform.float())
    return mel.clamp(min=1e-5).log()


class AudioMelConditioner(Conditioner):
    """Released 256-bin mel prompt projected to 768 dimensions."""

    def __init__(
        self,
        output_dim: int,
        mel_spec_type: str = "mel_features",
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mel_channels: int = 100,
        target_sample_rate: int = 24000,
        mask_ratio_start: float = 0.7,
        mask_ratio_end: float = 1.0,
        project_out: bool = False,
    ):
        if mel_spec_type != "mel_features":
            raise ValueError(f"unsupported MAF mel type {mel_spec_type!r}")
        super().__init__(768, output_dim, project_out=project_out)
        self.mel_spec_type = mel_spec_type
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_end = mask_ratio_end
        self.proj_features = nn.Linear(n_mel_channels, 768)

    def mask_mel_spectrogram(self, mels: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len = mels.shape
        ratios = torch.rand(batch_size, device=mels.device)
        ratios = ratios * (self.mask_ratio_end - self.mask_ratio_start) + self.mask_ratio_start
        masked = mels.clone()
        for index in range(batch_size):
            mask_len = int(seq_len * ratios[index])
            start = torch.randint(0, seq_len - mask_len + 1, (1,), device=mels.device)
            masked[index, :, start : start + mask_len] = 0
        return masked

    def forward(self, wavs, device):
        wavs = torch.cat(wavs, dim=0).to(device).float()
        mels = get_mel_spectrogram(
            wavs,
            n_fft=self.n_fft,
            n_mel_channels=self.n_mel_channels,
            target_sample_rate=self.target_sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        if self.mask_ratio_start < self.mask_ratio_end:
            mels = self.mask_mel_spectrogram(mels)
        embeddings = self.proj_features(mels.transpose(1, 2))
        return embeddings, torch.ones(embeddings.shape[0], 1, device=device)


def make_maf_conditioner(spec) -> MultiConditioner:
    """Build the three released AudioX-MAF conditioners."""
    config = spec.get_conditioning_dict()
    entries = {item["id"]: item for item in config["configs"]}
    output_dim = config["cond_dim"]
    video = MAFCLIPConditioner(output_dim=output_dim, **entries["video_prompt"]["config"])
    text = T5Conditioner(output_dim=output_dim, **entries["text_prompt"]["config"])
    audio = AudioMelConditioner(output_dim=output_dim, **entries["audio_prompt"]["config"])
    return MultiConditioner({"video_prompt": video, "text_prompt": text, "audio_prompt": audio})
