# Audio-GAR

Audio-GAR research codebase for audio generation tasks. This repo contains the finetuning and evaluation code tools.

# Abstract
Latent audio generative models are typically trained in two stages: an audio codec is learned first, followed by a latent generative model. This decomposition leads to a decoder train-generation mismatch: the codec decoder is trained on encoder-induced latents but deployed on generator-produced latents at inference time. Across diverse datasets and latent generative models, we observe clear reconstruction-generation gaps under both FD and FAD, showing that strong reconstruction quality does not necessarily translate into strong end-to-end generation quality. A natural remedy is to adapt the decoder on generation-produced latents, but generated latents lack correspondence with source audio and therefore cannot directly provide the paired supervision used for decoder fine-tuning. We introduce **AudioGAR**, which constructs intermediate latents by perturbing encoder latents and denoising them through the frozen latent diffusion model. These latents form a trajectory from reconstruction toward generation, with lower-noise latents retaining source correspondence and supporting paired decoder fine-tuning. We fine-tune only the codec decoder on these latents, while keeping the codec encoder and latent generative model frozen. When applied to AudioX, AudioGAR substantially improves generative performance. It requires only 1.5\% of the original training audio hours and 0.26\% of the original training cost.

## Results

Comparison with AudioX-MAF and TangoMusic on MusicCaps and AudioCaps. Arrows indicate the preferred direction.

| Model | Dataset | gFAD ↓ | gFD ↓ | KL ↓ | IS ↑ | PC ↑ | PQ ↑ |
|---|---|---|---|---|---|---|---|
| AudioX | MusicCaps | 1.60 | 9.56 | 1.00 | 3.65 | 4.78 | 6.61 |
| **AudioGAR AudioX** | MusicCaps | 1.10 | 8.33 | 0.99 | 3.59 | 4.75 | 6.56 |
| AudioX | AudioCaps | 1.61 | 11.83 | 1.31 | 12.47 | 3.16 | 5.73 |
| **AudioGAR AudioX** | AudioCaps | 1.34 | 11.62 | 1.29 | 12.06 | 3.11 | 5.73 |
| TangoMusic | MusicCaps | 1.85 | 15.20 | 1.09 | 2.85 | 5.62 | 7.22 |
| **AudioGAR TangoMusic** | MusicCaps | 1.47 | 14.25 | 1.09 | 2.83 | 5.61 | 7.16 |

## Models and eval cache

- Decoder weights: [overfittingexpert/Audio-GAR](https://huggingface.co/overfittingexpert/Audio-GAR)
  ([audiox-maf](https://huggingface.co/overfittingexpert/Audio-GAR/tree/main/audiox-maf),
  [tango-music](https://huggingface.co/overfittingexpert/Audio-GAR/tree/main/tango-music))
- Eval dataset (cached latent): [eval_cache](https://huggingface.co/overfittingexpert/Audio-GAR/tree/main/eval_cache)

## Setup

The code runs as a package named `audio_gar`, so clone it under that name.

```bash
git clone <repo-url> audio_gar
uv sync --project audio_gar
```

## Evaluation data

Metric weights and the pretrained AudioX-MAF come from their public releases:

```bash
mkdir -p data/fad_checkpoints
curl -L -o data/fad_checkpoints/fad_panns_cnn14_pytorch.pth \
  "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth"
curl -L -o data/fad_checkpoints/fad_vggish_pytorch.pth \
  https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth
curl -L -o data/fad_checkpoints/audiobox_aesthetics_checkpoint.pt \
  https://dl.fbaipublicfiles.com/audiobox-aesthetics/checkpoint.pt
uv run --no-sync --project audio_gar hf download HKUSTAudio/AudioX-MAF model.ckpt --local-dir data/audiox-maf
```

Reference audio is not redistributed. Place the AudioCaps and MusicCaps test clips under
`data/eval_datasets/<dataset>/` with a `manifest.csv` holding one row per caption:

## Getting started

```bash
# Fine-tuned decoder and eval cache from the Hub.
uv run --no-sync --project audio_gar hf download overfittingexpert/Audio-GAR \
  --include "audiox-maf/n0.2/decoder_100k.pt" "eval_cache/audiox-maf/musiccaps/*" \
  --local-dir hub

# 1. Latent cache: one-step denoised generator latents at noise level 0.2.
uv run --no-sync --project audio_gar python -m audio_gar.exp_latent_cache \
  --generator audiox-maf --noise_levels 0.2 \
  --data_root data --out_dir caches/audiox-maf

# 2. Decoder fine-tuning on the cached latents.
uv run --no-sync --project audio_gar python -m audio_gar.exp_train_decoder finetune \
  --cache_dirs caches/audiox-maf --level 0.2 --max_train_steps 100000 \
  --data_root data --out_dir runs/audiox-maf-n0.2

# 3. TangoMusic: cache at noise level 0.1, then joint decoder and vocoder training.
uv run --no-sync --project audio_gar python -m audio_gar.exp_latent_cache \
  --generator tango-music-af-ft-mc --noise_levels 0.1 \
  --data_root data --out_dir caches/tango-music
uv run --no-sync --project audio_gar python -m audio_gar.exp_train_decoder joint \
  --cache_dirs caches/tango-music --level 0.1 --max_train_steps 30000 \
  --data_root data --out_dir runs/tango-music-n0.1

# 4. Generation eval with the released decoder and eval cache.
uv run --no-sync --project audio_gar python -m audio_gar.eval \
  --model audiox-maf --dataset musiccaps --evaluation gen_ref \
  --base_ckpt data/audiox-maf/model.ckpt \
  --decoder_ckpt hub/audiox-maf/n0.2/decoder_100k.pt \
  --generation_cache hub/eval_cache/audiox-maf/musiccaps --cache_artifact_dir art \
  --data_root data --out_dir results/audiox-maf-musiccaps
```

## Citation

```bibtex
@inproceedings{FangAudiogar2026,
  title     = {},
  author    = {},
  booktitle = {},
  year      = {}
}
```
