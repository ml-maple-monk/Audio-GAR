import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch


# Hub shards pinned to one revision, metadata included.
@dataclass(frozen=True)
class DataShards:
    repo: str = ""
    revision: str = ""
    split: str = "train"
    shards: int = 0
    rows: int = 0
    text_fields: tuple = ()
    tag_fields: tuple = ()
    audio: str = "audio"
    gated: bool = False
    manifest: str = ""
    prompt_typed_ratio: float = 0.0


# Local audio tree joined to a sidecar metadata dump.
@dataclass(frozen=True)
class DataFolder:
    root: str = ""
    sidecar: str = ""
    sidecar_kind: str = ""
    manifest: str = ""
    suffix: str = ""
    nest: int = 0
    prompt_typed_ratio: float = 0.0

    # Audio path from an id, honouring the fma nesting.
    def get_path(self, data_root: Path, ident: str):
        base = data_root / self.root
        file_name = f"{ident}{self.suffix}"
        path = base / file_name
        if self.nest:
            path = base / ident[: self.nest] / file_name
        return path


# User prose, so a caption floor earns its keep.
@dataclass(frozen=True)
class DataFreesoundLaionTrain(DataShards):
    repo: str = "benjamin-paine/freesound-laion-640k"
    revision: str = "89e4b1035f3adf1c19f015a5ac274cc727f19992"
    shards: int = 1352
    rows: int = 455019
    text_fields: tuple = ("title", "description")
    tag_fields: tuple = ("tags",)


# Cluster-resident audio, one file per id.
@dataclass(frozen=True)
class DataFreesoundRaw(DataFolder):
    root: str = "audio_codec/freesound_ogg"
    sidecar: str = "metadata/freesound_metadata.jsonl"
    sidecar_kind: str = "freesound"
    suffix: str = ".ogg"


# fma_large nests a track under three leading digits.
@dataclass(frozen=True)
class DataFmaLarge(DataFolder):
    root: str = "audio_codec/fma/fma_large/fma_large"
    sidecar: str = "metadata/fma_metadata"
    sidecar_kind: str = "fma"
    suffix: str = ".mp3"
    nest: int = 3
    prompt_typed_ratio: float = 0.5


# Extracted audio joined to both caption manifests.
@dataclass(frozen=True)
class DataAudioxIfcapsTrain(DataFolder):
    root: str = "audioX_ifcap-data/audio"
    sidecar: str = "audioX_ifcap-data/metadata"
    sidecar_kind: str = "ifcaps"
    manifest: str = "audioX_ifcap-data/cache_manifest"
    suffix: str = ".mp3"
    repo: str = "HKUSTAudio/AudioX-IFcaps"
    revision: str = "3cb4d5295b170d35801e900f3442069ca0f8a176"
    text_fields: tuple = ("caption",)
    gated: bool = True


# Locally extracted WavCaps audio and prepared captions.
@dataclass(frozen=True)
class DataWavcapsTrain(DataFolder):
    root: str = "wavcaps_v1/audio"
    manifest: str = "wavcaps_v1/cache_manifest"
    suffix: str = ".flac"
    repo: str = "cvssp/WavCaps"
    revision: str = "0930ec11ded28fa0eaa910fde2f6fc3538acbeac"


# Locally staged AudioCaps train audio and captions.
@dataclass(frozen=True)
class DataAudiocapsTrain(DataFolder):
    root: str = "audiocaps_train_v1/audio"
    manifest: str = "audiocaps_train_v1/cache_manifest"
    suffix: str = ".wav"
    repo: str = "d0rj/audiocaps"
    revision: str = "54887eb2a01bf806cdbec0aca41fd85628dac0e4"
    split: str = "train"


# AudioTime train audio with GPT-generated captions.
@dataclass(frozen=True)
class DataAudiotimeTrain(DataFolder):
    root: str = "audiotime_train_v1/audio"
    manifest: str = "audiotime_train_v1/cache_manifest"
    suffix: str = ".wav"
    repo: str = "zeyuxie29/AudioTime"
    revision: str = "9ac2023049c00e3e8375baab8145a6177cba4501"
    split: str = "train"


# OpenSound parquet mirror of the audiocaps train split.
@dataclass(frozen=True)
class DataAudiocapsHfTrain(DataShards):
    repo: str = "OpenSound/AudioCaps"
    revision: str = "b29b3243d6ce49c2cd0d48d4b5f0701ae7969ded"
    shards: int = 412
    text_fields: tuple = ("caption",)


# Extracted audio joined to one captions json.
@dataclass(frozen=True)
class DataCaptionFolder(DataFolder):
    suffix: str = ".flac"
    sidecar_kind: str = "captions"
    repo: str = ""
    revision: str = ""


# One WavCaps domain folder at the pinned hub revision.
@dataclass(frozen=True)
class DataWavcapsCaptions(DataCaptionFolder):
    repo: str = "cvssp/WavCaps"
    revision: str = "0930ec11ded28fa0eaa910fde2f6fc3538acbeac"


# One MusicCaps split folder at the pinned hub revision.
@dataclass(frozen=True)
class DataMusiccapsCaptions(DataCaptionFolder):
    repo: str = "agkphysics/AudioSet"
    revision: str = "0c609e8302cf139307f639c57652032af0a88041"


# Explicit audio-and-caption rows from one jsonl listing.
@dataclass(frozen=True)
class DataClipList:
    listing: str = ""
    prompt_typed_ratio: float = 0.0

    # Manifest audio paths resolve against the data root.
    def get_path(self, data_root: Path, ident: str):
        path = Path(ident)
        if not path.is_absolute():
            path = data_root / path
        return path


# Member datasets one mix name walks round robin.
@dataclass(frozen=True)
class DataMix:
    members: tuple = ()


VERSIONS: dict = {
    "freesound_laion_train_v1": DataFreesoundLaionTrain(),
    "freesound_raw_v1": DataFreesoundRaw(),
    "fma_large_v1": DataFmaLarge(),
    "audiox_ifcaps_train_v1": DataAudioxIfcapsTrain(),
    "wavcaps_train_v1": DataWavcapsTrain(),
    "audiocaps_train_v1": DataAudiocapsTrain(),
    "audiotime_train_v1": DataAudiotimeTrain(),
    "wavcaps_audioset_v1": DataWavcapsCaptions(
        root="wavcaps_sub_v1/AudioSet_SL",
        sidecar="wavcaps_sub_v1/json_files/AudioSet_SL/as_final.json",
    ),
    "wavcaps_bbc_v1": DataWavcapsCaptions(
        root="wavcaps_sub_v1/BBC_Sound_Effects",
        sidecar="wavcaps_sub_v1/json_files/BBC_Sound_Effects/bbc_final.json",
    ),
    "wavcaps_freesound_v1": DataWavcapsCaptions(
        root="wavcaps_sub_v1/FreeSound",
        sidecar="wavcaps_sub_v1/json_files/FreeSound/fsd_final.json",
    ),
    "wavcaps_soundbible_v1": DataWavcapsCaptions(
        root="wavcaps_sub_v1/SoundBible",
        sidecar="wavcaps_sub_v1/json_files/SoundBible/sb_final.json",
    ),
    "audiocaps_train_hf_v1": DataAudiocapsHfTrain(),
    "musiccaps_train_v1": DataMusiccapsCaptions(
        root="musiccaps_sub_v1/MusicCaps_train",
        sidecar="musiccaps_sub_v1/json_files/MusicCaps_train/mc_train_final.json",
    ),
    "musiccaps_eval_v1": DataMusiccapsCaptions(
        root="musiccaps_sub_v1/MusicCaps_eval",
        sidecar="musiccaps_sub_v1/json_files/MusicCaps_eval/mc_eval_final.json",
    ),
    "musiccaps_48k_v1": DataFolder(root="musiccaps_48k/audio", suffix=".flac"),
    "bulk_v1_audio": DataFolder(root="bulk_v1/audio", suffix=".flac"),
    "deficit_nn_audio": DataFolder(root="deficit_nn/audio", suffix=".flac"),
    "deficit_cls_audio": DataFolder(root="deficit_cls/audio", suffix=".flac"),
    "deficit_v1_audio": DataFolder(root="deficit_v1/audio", suffix=".flac"),
}


DATASET_MIXES: dict = {
    "wavcaps_all_v1": DataMix(members=(
        ("wavcaps_audioset_v1", 1.0),
        ("wavcaps_bbc_v1", 1.0),
        ("wavcaps_freesound_v1", 1.0),
        ("wavcaps_soundbible_v1", 1.0),
    )),
}


# Shard sizes and prompt columns for raw dataset walks.
@dataclass(frozen=True)
class DataWalkConfig:
    folder_shard: int = 512
    parquet_batch: int = 16
    # Stable Audio Open 3.2 names these five FMA metadata types.
    fma_column_list: tuple = (
        ("year", ("album", "date_released")),
        ("genres", ("track", "genres")),
        ("album", ("album", "title")),
        ("title", ("track", "title")),
        ("artist", ("artist", "name")),
    )
    # The paper names description, title and tags for freesound.
    freesound_column_list: tuple = (("title", "name"), ("description", "description"))


@dataclass(frozen=True)
class DataManifest:
    digest: str
    clips: int
    shards: int
    rank_shards: tuple


# Clips one shard walk kept, and how many it dropped.
@dataclass
class DataClipScan:
    clip_list: list
    dropped: int


# Per-dataset state of one round-robin walk over shards.
@dataclass
class DataWalk:
    name_list: list
    spec_dict: dict
    budget_dict: dict
    limit_dict: dict
    kept_dict: dict
    dropped_dict: dict
    passed_dict: dict


# Eval-set ytids the finetunes must never train on.
def load_eval_exclusions(data_root: Path):
    path = data_root / "wavcaps_v1" / "eval_exclusions.jsonl"
    text = path.read_text()
    ytid_list = []
    for line_item in text.splitlines():
        if line_item:
            row = json.loads(line_item)
            ytid_list.append(row["ytid"])
    exclusions = frozenset(ytid_list)
    return exclusions


# Pinned dataset spec for one version id.
def get_version(version: str):
    if version.startswith("list:"):
        part_tuple = version.partition(":")
        spec = DataClipList(listing=part_tuple[2])
    elif version in VERSIONS:
        spec = VERSIONS[version]
    else:
        spec = DATASET_MIXES[version]
    return spec


@dataclass(frozen=True)
class SampleClip:
    dataset: str
    sample_hash: str
    source: str
    metadata: tuple
    # probed source duration; zero means never probed
    seconds: float = 0.0


# Locator split into dataset, shard or file mark, and id.
@dataclass(frozen=True)
class DataLocatorParts:
    dataset: str
    middle: str
    ident: str


# Locators and identities of clips inside pinned datasets.
class DataLocator:
    # Row address inside a pinned shard, stable across runs.
    def make_locator(self, dataset: str, shard: int, row: int):
        locator = f"{dataset}#{shard:05d}#{row:05d}"
        return locator

    # Folder rows address the file itself, so lookup is direct.
    def make_file_locator(self, dataset: str, ident: str):
        locator = f"{dataset}#file#{ident}"
        return locator

    # Split one locator into its three hash-separated parts.
    def parse_locator(self, source: str):
        dataset, middle, ident = source.split("#")
        parts = DataLocatorParts(dataset=dataset, middle=middle, ident=ident)
        return parts

    # Folder clip identity from dataset, id and byte count.
    def make_file_hash(self, dataset: str, ident: str, size: int):
        key = f"{dataset}/{ident}:{size}"
        key_bytes = key.encode()
        sha = hashlib.sha1(key_bytes)
        file_hash = sha.hexdigest()
        return file_hash

    # List clip identity; caption edits re-key the row.
    def make_list_hash(self, dataset: str, ident: str, size: int, caption: str):
        key = f"{dataset}/{ident}:{size}:{caption}"
        key_bytes = key.encode()
        blake = hashlib.blake2b(key_bytes, digest_size=20)
        list_hash = blake.hexdigest()
        return list_hash


# Dataset name carried by the source locator.
def get_domain(source: str):
    part_list = source.split("#", 1)
    domain = part_list[0]
    return domain


prompt_policy = "sao_metadata_fma_typed_v1"


# caption text reaches the encoder as written
VERBATIM_POLICY = "caption_verbatim_v1"


LATENT_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


# Latent cache CLI defaults, one field per option.
@dataclass(frozen=True)
class CacheRunConfig:
    data_root: Path = Path("")
    out_dir: Path = Path("")
    generator: str = "audiox-maf"
    corpus: str = "freesound_raw_v1,freesound_laion_train_v1,fma_large_v1"
    max_files: int = 1000000
    skip_files: int = 0
    chars_min: int = 12
    clip_seconds: float = 10.0
    cond_policy: str = "window"
    crop_policy: str = "head"
    crop_margin: float = 0.0
    channel_policy: str = "stereo"
    volume_norm: bool = False
    peak_norm: bool = False
    short_policy: str = "repeat"
    prompt_policy: str = prompt_policy
    noise_levels: str = "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
    noise_schedule: str = "cosine"
    shard_clips: int = 256
    batch_clips: int = 8
    exec_batch_clips: int = 0
    max_shards: int = 0
    num_steps: int = 0
    cfg_scale: float = 7.0
    store_dtype: str = "float16"
    determinism: bool = False
    vae_dtype: str = "bfloat16"
    gen_dtype: str = "bfloat16"
    compile_blocks: bool = False
    compile_mode: str = "auto"
    seed: int = 0
    device: str = "cuda"


# Latent cache output layout under --out_dir.
@dataclass(frozen=True)
class CacheWriterConfig:
    artifact_dir: str = "art"
    default_ext: str = ".json"
    progress_name: str = "progress.jsonl"
    summary_name: str = "summary.json"
    latent_shard: str = ".safetensors"
    latent_clip: str = ".jsonl"
    latent_level: str = ".jsonl"


# Train output layout; art/ keeps the kernel checkpoint names.
@dataclass(frozen=True)
class TrainWriterConfig:
    artifact_dir: str = "art"
    debug_dir: str = "debug"
    default_ext: str = ".json"
    progress_name: str = "progress.jsonl"
    summary_name: str = "summary.json"
    decoder_ckpt: str = ".ckpt"
    vae_ckpt: str = ".ckpt"
    train_tracking: str = ".sqlite"
    # kernel Run seed offset; old draws stay reproducible
    item_seed_offset: int = 100_000


# Frozen cache protocol values, never exposed as options.
@dataclass(frozen=True)
class CacheDrawConfig:
    # ladder winner for audiox (archived/torchcompile_audiox_logs)
    auto_compile_mode: str = "cudagraph"
    # distinct stream for policy picks, offsets keep the crop stream
    pick_salt: int = 0x517CC1B727220A95
    # distinct stream for loudness draws, crop stream untouched
    gain_salt: int = 0x2545F4914F6CDD1D
    lufs_target: float = -16.0
    lufs_jitter: float = 2.0
    silence_dbmax: float = -60.0
    energy_floor: float = 1e-6


# Cosine schedule (a_t, b_t), unit L2 norm.
def get_cosine_noise_coefficients(noise_level: float):
    norm = math.hypot(1.0 - noise_level, noise_level)
    a_t = (1.0 - noise_level) / norm
    b_t = noise_level / norm
    return a_t, b_t


# Steps track t, so every level walks one shared stride.
def get_schedule_steps(num_steps: int, noise_level: float):
    a_t, b_t = get_cosine_noise_coefficients(noise_level)
    steps = 0
    if b_t != 0.0:
        angle = math.atan2(b_t, a_t)
        t_model = angle * 2.0 / math.pi
        scaled_steps = round(num_steps * t_model)
        steps = max(1, scaled_steps)
    return steps


# Schedule rows shared by cache producer and reader.
def make_level_rows(levels, num_steps: int, steps_fn=None, coeffs_fn=None):
    step_fn = get_schedule_steps
    if steps_fn:
        step_fn = steps_fn
    coeff_fn = get_cosine_noise_coefficients
    if coeffs_fn:
        coeff_fn = coeffs_fn
    row_list = []
    for level_item in levels:
        a_t, b_t = coeff_fn(level_item)
        angle = math.atan2(b_t, a_t)
        t_model = angle * 2.0 / math.pi
        steps = step_fn(num_steps, level_item)
        row = {
            "noise_level": level_item, "a_t": a_t, "b_t": b_t,
            "t_model": t_model, "steps": steps,
        }
        row_list.append(row)
    return row_list


@dataclass(frozen=True)
class SampleLatent:
    cache_dir: str
    shard: int
    row: int
    level: str
    source: str
    valid_frames: int
    gain: float = 1.0
    phase_flip: bool = False
    start_sample: int = 0


# Drop counts and kept latents of one cache scan.
@dataclass
class LatentScan:
    name: str
    latent_list: list
    shards: int
    corrupted_wav: int
    short: int
    taken: int


# One aligned latent and waveform crop.
@dataclass(frozen=True)
class LatentCrop:
    latent: torch.Tensor
    wav: torch.Tensor


# Header artifact of one latent cache slot.
def get_cache_header_path(cache_dir: Path):
    header_path = cache_dir / "art" / "latent_cache.json"
    return header_path


# Latent blob of one shard, legacy .bin honoured.
def get_cache_shard_path(cache_dir: Path, shard: int, level: str):
    name = f"latent_shard.shard{shard:06d}_{level}"
    legacy = cache_dir / "art" / f"{name}.bin"
    shard_path = cache_dir / "art" / f"{name}.safetensors"
    if legacy.is_file():
        shard_path = legacy
    return shard_path


# Every latent blob of one level label, sorted.
def get_cache_shard_paths(cache_dir: Path, level: str):
    art_dir = cache_dir / "art"
    found = {}
    for suffix_item in ("safetensors", "bin"):
        for path_item in art_dir.glob(f"latent_shard.shard*_{level}.{suffix_item}"):
            part_list = path_item.name.rsplit(".", 1)
            found[part_list[0]] = path_item
    path_list = []
    for name_item in sorted(found):
        path_list.append(found[name_item])
    return path_list


# Every per-shard clip table, sorted.
def get_cache_clip_paths(cache_dir: Path):
    art_dir = cache_dir / "art"
    clip_paths = art_dir.glob("latent_clip.shard*.jsonl")
    path_list = sorted(clip_paths)
    return path_list


# Decoder input and ground truth waveform.
@dataclass(frozen=True)
class VaeBatch:
    latent: torch.Tensor
    ground_truth: torch.Tensor

    # DataLoader pin hook, so device copies overlap.
    def pin_memory(self):
        latent = self.latent.pin_memory()              # (B, D, F)
        ground_truth = self.ground_truth.pin_memory()
        pinned = VaeBatch(latent=latent, ground_truth=ground_truth)
        return pinned


# Crop geometry and prefetch bounds for decoder training.
@dataclass(frozen=True)
class SpecTrainData:
    batch_clips: int
    accumulation: int
    crop_frames: int
    sample_size: int
    data_workers: int
    data_prefetch: int
    peak_norm: bool = False


@dataclass(frozen=True)
class SampleEval:
    dataset: str
    sample_id: str
    recording_id: str
    caption: str
    audio_path: Path | None
    start_seconds: float
    duration_seconds: float
    subset: str
    sample_rate: int | None = None


@dataclass(frozen=True)
class AudioProbeResult:
    sample_rate: int
    num_frames: int
