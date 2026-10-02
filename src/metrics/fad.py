from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from audiogen_eval_protocols import FadSpec, PannSpec
import numpy as np
import torch
from scipy import linalg
from torchaudio.functional import resample
from torchvggish import waveform_to_examples
from torchvggish.torchvggish import _vgg


@dataclass
class FadStats:
    mu: np.ndarray
    sigma: np.ndarray


@dataclass
class PannRowsLogits:
    row_array: np.ndarray
    logit_array: np.ndarray


@dataclass
class FadRows:
    row_array: np.ndarray
    count_array: np.ndarray


class FadDistance:
    def __init__(self, fd_eps: float):
        self.fd_eps = fd_eps

    # Mean and covariance of one embedding set.
    def get_stats(self, embd_array):
        embd = np.asarray(embd_array)
        mu = np.mean(embd, axis=0)
        sigma = np.cov(embd, rowvar=False)
        stats = FadStats(mu=mu, sigma=sigma)
        return stats

    # Numpy Frechet distance, after pytorch-fid and audioldm_eval.
    def get_frechet_distance(self, ref_stats: FadStats, gen_stats: FadStats):
        mu_ref = np.atleast_1d(ref_stats.mu)
        mu_gen = np.atleast_1d(gen_stats.mu)
        sigma_ref = np.atleast_2d(ref_stats.sigma)
        sigma_gen = np.atleast_2d(gen_stats.sigma)
        diff = mu_ref - mu_gen
        product = sigma_ref.dot(sigma_gen)
        covmean = linalg.sqrtm(product)
        # scipy 1.18 dropped disp; audioldm_eval keeps the offset retry
        if not np.isfinite(covmean).all():
            print(f"fd covariance product is singular; retrying with eps {self.fd_eps}")
            offset = np.eye(sigma_ref.shape[0]) * self.fd_eps
            shifted = (sigma_ref + offset).dot(sigma_gen + offset)
            covmean = linalg.sqrtm(shifted)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        tr_covmean = np.trace(covmean)
        fd = diff.dot(diff) + np.trace(sigma_ref) + np.trace(sigma_gen) - 2 * tr_covmean
        fd_value = float(fd)
        return fd_value

    # Frechet distance between two embedding sets.
    def get_fad(self, ref_array, gen_array):
        ref_stats = self.get_stats(ref_array)
        gen_stats = self.get_stats(gen_array)
        fd_value = self.get_frechet_distance(ref_stats, gen_stats)
        return fd_value

    # Paired AudioSet softmax KL.
    def get_kl_softmax(self, ref_logit_array, gen_logit_array, kl_eps: float):
        ref_array = np.asarray(ref_logit_array)
        gen_array = np.asarray(gen_logit_array)
        target = torch.from_numpy(ref_array)
        predicted = torch.from_numpy(gen_array)
        softmax_log = (predicted.softmax(dim=1) + kl_eps).log()
        softmax_target = target.softmax(dim=1)
        softmax_sum = torch.nn.functional.kl_div(softmax_log, softmax_target, reduction="sum")
        kl_value = float(softmax_sum / len(predicted))
        return kl_value

    # Mean inception score over generated AudioSet logits.
    def get_inception_mean(self, gen_logit_array, splits: int, seed: int):
        logit_array = np.asarray(gen_logit_array)
        features = torch.from_numpy(logit_array).double()
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(features))
        features = features[order]
        probabilities = features.softmax(dim=1)
        log_probabilities = features.log_softmax(dim=1)
        score_list = []
        total = len(features)
        for split_idx in range(splits):
            low = split_idx * total // splits
            high = (split_idx + 1) * total // splits
            part = probabilities[low:high]
            log_part = log_probabilities[low:high]
            marginal = part.mean(dim=0, keepdim=True)
            divergence = part * (log_part - marginal.log())
            split_score = divergence.sum(dim=1).mean().exp()
            score_list.append(float(split_score))
        is_mean = float(np.mean(score_list))
        return is_mean


class PannEmbedder:
    def __init__(self, model, spec: PannSpec, device: str):
        self.model = model
        self.spec = spec
        self.device = device

    # Load CNN14 with the spec's checkpoint.
    @classmethod
    def make_from_ckpt(cls, ckpt_dir: Path, device: str, spec: PannSpec):
        # panns_inference stays lazy so other backbones skip its import
        panns_models = import_module("panns_inference.models")
        cnn14 = panns_models.Cnn14
        model = cnn14(
            sample_rate=spec.sample_rate,
            window_size=spec.window_size,
            hop_size=spec.hop_size,
            mel_bins=spec.mel_bins,
            fmin=spec.fmin,
            fmax=spec.fmax,
            classes_num=spec.classes,
        )
        ckpt_path = Path(ckpt_dir) / spec.checkpoint_name
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()
        embedder = cls(model, spec, device)
        return embedder

    # Downmix to mono float audio on the device.
    def get_mono_audio(self, audio: torch.Tensor):
        mono = audio
        if audio.ndim == 2:
            mono = audio.mean(dim=0)
        mono = mono.detach().to(device=self.device, dtype=torch.float32)
        return mono

    # Downmix and resample one clip into a single CNN14 input.
    def get_chunks(self, audio: torch.Tensor, sample_rate: int, remove_dc: bool,
                   pad_samples: int):
        mono = self.get_mono_audio(audio)
        if sample_rate != self.spec.sample_rate:
            mono = resample(mono, sample_rate, self.spec.sample_rate)
        if remove_dc:
            mono = mono - mono.mean()
        # zero keeps true length; one STFT window minimum
        floor = pad_samples or self.spec.window_size
        if mono.numel() < floor:
            mono = torch.nn.functional.pad(mono, (0, floor - mono.numel()))
        chunks = mono.unsqueeze(0)
        return chunks

    # CNN14 embeddings, one finite row per clip.
    @torch.no_grad()
    def get_embeddings(self, audio_iter, sample_rate: int, remove_dc: bool, pad_samples: int):
        embedding_list = []
        for audio_item in audio_iter:
            chunks = self.get_chunks(audio_item, sample_rate, remove_dc, pad_samples)
            output = self.model(chunks)
            clip_array = output["embedding"].detach().cpu().numpy()
            finite_mask = np.isfinite(clip_array).all(axis=1)
            embedding_list.append(clip_array[finite_mask])
        row_array = np.concatenate(embedding_list, axis=0)
        return row_array

    # Paired CNN14 embedding and logit rows, bf16 forward.
    @torch.no_grad()
    def get_rows_logits(self, audio_iter, sample_rate: int, remove_dc: bool, pad_samples: int):
        device_type = torch.device(self.device).type
        row_list = []
        logit_list = []
        for audio_item in audio_iter:
            chunks = self.get_chunks(audio_item, sample_rate, remove_dc, pad_samples)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                output = self.model(chunks)
                embedding = output["embedding"]
                logits = self.model.fc_audioset(embedding)
            row_list.append(embedding.float().cpu().numpy())
            logit_list.append(logits.float().cpu().numpy())
        result = PannRowsLogits(
            row_array=np.concatenate(row_list, axis=0),
            logit_array=np.concatenate(logit_list, axis=0),
        )
        return result

    # AudioSet logits before the sigmoid, one row per clip.
    @torch.no_grad()
    def get_logits(self, audio_iter, sample_rate: int):
        logit_list = []
        for audio_item in audio_iter:
            clip = self.get_mono_audio(audio_item)
            if sample_rate != self.spec.sample_rate:
                clip = resample(clip, sample_rate, self.spec.sample_rate)
            if clip.numel() < self.spec.window_samples:
                pad_size = self.spec.window_samples - clip.numel()
                clip = torch.nn.functional.pad(clip, (0, pad_size))
            # in eval the embedding is the pre-logit activation
            output = self.model(clip.unsqueeze(0))
            logits = self.model.fc_audioset(output["embedding"])
            logit_list.append(logits.detach().cpu().numpy())
        logit_array = np.concatenate(logit_list, axis=0)
        logit_array = logit_array.astype(np.float64)
        return logit_array


class FadEmbedder:
    def __init__(self, model, spec: FadSpec, device: str):
        self.model = model
        self.spec = spec
        self.device = device

    # Load torchvggish from the spec's checkpoint.
    @classmethod
    def make_from_ckpt(cls, ckpt_dir: Path, device: str, spec: FadSpec):
        model = _vgg(postprocess=False)
        ckpt_path = Path(ckpt_dir) / spec.checkpoint_name
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        if not spec.use_activation:
            layer_list = list(model.embeddings.children())
            model.embeddings = torch.nn.Sequential(*layer_list[:-1])
        model.to(device)
        model.eval()
        embedder = cls(model, spec, device)
        return embedder

    # FAD rows plus each clip's row count.
    @torch.no_grad()
    def get_rows(self, audio_iter, sample_rate: int, remove_dc: bool):
        embedding_list = []
        count_list = []
        for audio_item in audio_iter:
            mono = audio_item
            if audio_item.ndim == 2:
                mono = audio_item.mean(dim=0)
            mono = mono.detach().to(device="cpu", dtype=torch.float32)
            if sample_rate != self.spec.sample_rate:
                mono = resample(mono, sample_rate, self.spec.sample_rate)
            if remove_dc:
                mono = mono - mono.mean()
            examples = waveform_to_examples(mono.numpy(), self.spec.sample_rate)
            examples = examples.to(self.device)
            output = self.model(examples)
            output_array = output.detach().cpu().numpy().astype(np.float64)
            clip_array = np.atleast_2d(output_array)
            finite_mask = np.isfinite(clip_array).all(axis=1)
            embedding_list.append(clip_array[finite_mask])
            count_list.append(int(finite_mask.sum()))
        result = FadRows(
            row_array=np.concatenate(embedding_list, axis=0),
            count_array=np.asarray(count_list, dtype=np.int64),
        )
        return result
