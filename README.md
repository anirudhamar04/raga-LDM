# Raga-LDM

This repo holds two things:

1. **`ragaldm/spleeter_torch/` + top-level `*_torch.py` scripts** — a PyTorch port of Deezer Spleeter trained on a 5-stem Carnatic dataset (`vocals`, `violin`, `ghatam`, `mridangam`, `drone`). This is the active code path; designed to run on A100 / modern CUDA via bf16 autocast.
2. **Audio processing pipeline** (`process_audio_pipeline.py`, `run_processing.py`, `audio_config.py`) for preparing Carnatic / Hindustani datasets for ML — described in the lower half of this README.

The original TensorFlow Spleeter training/inference code lives under `legacy/` for reference and one-time weight conversion. New work should use the PyTorch path.

## PyTorch 5-stem Spleeter: Quick start

### Install

```bash
uv sync                          # PyTorch + torchaudio + soundfile (no TF)
uv sync --extra convert          # also pulls TF 2.15 + the legacy/ spleeter package
                                 # (only needed for the one-time weight conversion)
```

`pyproject.toml` pins torch / torchaudio to the official PyTorch CUDA 12.8 index. Adjust the index URL there if your A100 host has a different CUDA driver.

### Convert pretrained Deezer weights → PyTorch `.pt` (one-time)

```bash
# Inspect what's in the TF checkpoint:
python convert_spleeter_weights.py \
    --pretrained spleeter:5stems \
    --target_config legacy/spleeter/configs/5stems_carnatic/base_config.json \
    --dry_run

# Real conversion:
python convert_spleeter_weights.py \
    --pretrained spleeter:5stems \
    --target_config legacy/spleeter/configs/5stems_carnatic/base_config.json \
    --output trained_models/pretrained_5stems_carnatic.pt
```

The converter:
- Downloads the Deezer pretrained checkpoint via `spleeter.model.provider.ModelProvider`.
- Maps source stems → 5 Carnatic stems (`piano→violin`, `bass→drone`, `drums→mridangam` AND `→ghatam`, `vocals→vocals`).
- Transposes Conv2D / Conv2DTranspose kernel layouts and copies BN running stats.
- Writes a self-contained `.pt` that the training script can warm-start from.

### Train

Epoch-based; one full pass over the training DataLoader per epoch. After each epoch:
- runs validation,
- writes `ckpt_epoch_{N}.pt` (last `--keep_last` kept; default 5),
- updates `latest.pt` (used by `--resume` and inference),
- updates `best.pt` whenever `val/loss` improves.

```bash
python train_spleeter_torch.py \
    --config configs/colab_5stems_carnatic.json \
    --data /content \
    --pretrained_weights trained_models/pretrained_4stems_carnatic.pt
```

Key flags: `--resume` (continue from `latest.pt`), `--max_epochs N`, `--keep_last N`, `--validate_only`, `--skip_validation`, `--no_wandb`, `--num_workers N`. Training uses `bf16` autocast on CUDA by default; pass `--no_bf16` for fp32. Optional config key `train_max_steps` is a hard step cap, useful for smoke tests.

### Running on Google Colab

1. Mount Drive and clone the repo into `/content`:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !git clone https://github.com/<your-fork>/raga-LDM /content/raga-LDM
   %cd /content/raga-LDM
   ```
2. Install deps. On a fresh Colab the base env already has torch, so use `uv pip install` for just what's missing, or `uv sync` for a clean env:
   ```python
   !pip install -q uv
   !uv sync                  # PyTorch base
   !uv sync --extra convert  # +TF, only for weight conversion
   ```
3. Convert pretrained weights once (cached to Drive so subsequent sessions skip the download):
   ```python
   import os; os.environ['MODEL_PATH'] = '/content/drive/MyDrive/Models/spleeter_cache'
   !uv run python convert_spleeter_weights.py \
       --pretrained spleeter:4stems \
       --target_config configs/colab_5stems_carnatic.json \
       --output /content/drive/MyDrive/Models/Spleeter/pretrained_4stems_carnatic.pt
   ```
4. Train. The `configs/colab_5stems_carnatic.json` config:
   - Reads `train_csv` / `validation_csv` at `/content/raga-LDM/example_dataset/{train,validation}.csv`.
   - Writes all checkpoints to `/content/drive/MyDrive/Models/Spleeter` (`ckpt_epoch_*.pt`, `latest.pt`, `best.pt`).
   - Uses `--data /content` so the `drive/MyDrive/...`-relative paths in the CSVs resolve to `/content/drive/MyDrive/...`.
   ```python
   !uv run python train_spleeter_torch.py \
       --config configs/colab_5stems_carnatic.json \
       --data /content \
       --pretrained_weights /content/drive/MyDrive/Models/Spleeter/pretrained_4stems_carnatic.pt \
       --num_workers 8
   ```
5. If the runtime disconnects, resume with the same command plus `--resume` — `latest.pt` carries the epoch counter, optimizer state, and best-val-loss across restarts.

A100 config notes (`configs/colab_5stems_carnatic.json`):
- `batch_size: 16` — fits comfortably in 40 GB under bf16 autocast (~20 GB activations + 600 MB Adam state + ~200 MB params). Bump to 32 on 80 GB if you want, but the example_dataset only has 30 train songs so larger batches give fewer optimizer steps per epoch — usually not worth it.
- `n_chunks_per_song: 4` — pulls four offset segments out of each song instead of two. With ~30 songs you get 120 chunks/epoch → ~7 batches at `batch_size=16`. Random T-frame crop on top of each segment gives further diversity.
- `max_epochs: 200` — ~1400 optimizer steps total on the example dataset; on A100 the whole run finishes in tens of minutes. Loss should be well-converged or starting to overfit (watch the gap between `train/loss` and `val/loss`).
- `learning_rate: 5e-5` — the conservative Spleeter fine-tune default. If `train/loss` is barely moving in the first 20 epochs, try `1e-4` (sqrt-scaling-rule equivalent for `batch_size=16` up from the original `batch_size=4`).
- `save_summary_steps: 25` — logs to console + wandb roughly every 3–4 epochs at this batch/dataset size.
- `--num_workers 8` — Colab A100 instances typically expose 12 vCPUs; 8 leaves a couple for the main process.

**Augmentation** (`augmentation` block of the config). Train-only; val always sees clean clips. Every augmentation is applied identically to the mix and every stem (sampled once per `__getitem__` call via an `_AugmentationPlan`), so the `mix = sum(stems)` waveform invariant is preserved — verified by `smoke_test/verify_aug_invariant.py`.

| Key | Default in Colab config | What it does |
| --- | --- | --- |
| `stitch_prob` | `0.4` | With this probability, pick a second chunk index `k_secondary != k` from the same song, load that chunk, and splice the two windows at a random sample with a 512-sample (~12 ms) linear crossfade. Forces the model to handle within-song temporal context shifts. Disabled automatically when `n_chunks_per_song == 1`. |
| `gain_prob` | `0.5` | With this probability, sample a single gain in `[-gain_db_max, +gain_db_max]` dB and multiply every waveform (mix + stems) by `10**(gain/20)`. Mild level robustness. |
| `gain_db_max` | `3.0` | Max abs gain in dB. ±3 dB ≈ factor 0.71–1.41. |
| `channel_swap_prob` | `0.5` | With this probability, swap left/right channels on every waveform (stereo only). |

To disable augmentation, either drop the `augmentation` block from the config or set every prob to `0.0`.

What's *not* in here (deliberately):
- **Gaussian noise on the mix** would break `mix = sum(stems)` unless we also add the same noise to a "noise" stem, which we don't have. Useful for denoising-style training, not source separation.
- **Time stretch / pitch shift** are linear when applied identically to mix + stems but require resampling on every load and are slow. Skip unless you find the model overfitting to timing/key.
- **SpecAugment (time + frequency masking)** masks the model *input* but not the targets — a different objective (robustness to occlusions) than what Spleeter is solving. Add it as a separate flag if you want.

Other things to know:
- Drive writes are slow; `torch.save` to `/content/drive/...` can take a few seconds per epoch. That's normal.
- If you're using `spleeter:5stems` pretrained instead of 4stems, change the config's `model.type` to `unet.softmax_unet` and re-convert with `--pretrained spleeter:5stems`. The current config uses `unet.unet` which matches `spleeter:4stems`.
- `--no_wandb` disables logging if you don't have wandb configured. Set `WANDB_ENTITY` / `WANDB_PROJECT` / `WANDB_API_KEY` env vars in the Colab notebook to customise (defaults are `RetinalDistill` / `CarnaticSpeeter`).

CSV format (paths relative to `--data`):

```
mix_path,vocals_path,violin_path,ghatam_path,mridangam_path,drone_path,duration
Concert04/06-Balagopala/.../mix.wav,.../vocals.wav,.../violin-1.wav,.../ghatam-close.wav,.../mridangam-left.wav,.../tanpura.wav,2675.4
```

### Inference

```bash
python infer_spleeter_torch.py \
    --model_dir trained_models/5stems_carnatic_pt \
    --input recording.wav \
    --output separated/

# Recursive over a directory:
python infer_spleeter_torch.py \
    --model_dir trained_models/5stems_carnatic_pt \
    --input_dir clips/ --recursive \
    --output separated/
```

Inference is always chunked (30 s default, frame-length overlap, linear crossfade) so memory is predictable on any-length input. Codec options: `wav`, `flac` (via soundfile), `mp3`, `ogg`, `m4a` (via `ffmpeg` on PATH).

### Layout

```
ragaldm/spleeter_torch/    # PyTorch package
    model.py               # UNet + Stem5UNet (5 per-stem U-Nets + power-law mask)
    audio.py               # STFT/iSTFT + mask-extend, byte-equivalent to legacy Spleeter
    dataset.py             # PyTorch Dataset for the Spleeter CSV format
    losses.py              # l1_mask_loss
convert_spleeter_weights.py  # TF -> PyTorch one-shot
train_spleeter_torch.py
infer_spleeter_torch.py
legacy/                    # original TF Spleeter stack, kept for weight conversion only
    spleeter/              # vendored Deezer Spleeter (TF 2.15)
    train_spleeter_5stems.py
    infer_spleeter_5stems.py
    utils/                 # init_pretrained_weights, training_helpers
```

---

# Audio Processing Pipeline (dataset prep)

The dataset prep pipeline (used to build the CSVs the trainer consumes) lives at the repo root: `process_audio_pipeline.py`, `run_processing.py`, `audio_config.py`.

## Features

The pipeline processes audio files through multiple stages:

1. **Mono Conversion** - Convert stereo to mono
2. **Loudness Normalization** - Normalize to -18 LUFS
3. **Silence Trimming** - Remove leading/trailing silence
4. **Chunking** - Split into 30-second segments
5. **Energy Filtering** - Remove low-energy/drone-only clips
6. **Raga Extraction** - Automatically extract raga information from dataset annotations
7. **Source Separation** - Using Demucs to separate:
   - Drums/Mridangam/Tabla (always)
   - Accompaniment (Violin/Tanpura/other instruments) (always)
   - Vocals (if detected)
8. **Metadata Generation** - Create detailed CSV with all processing information
9. **Checkpoint/Resume** - Automatically saves progress every 5 files, resume after interruption

## Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements_processing.txt
```

Or using uv:
```bash
uv pip install -r requirements.txt
uv pip install -r requirements_processing.txt
```

### Basic Usage

Process all audio files in the `data/` directory:

```bash
python run_processing.py
```

Or use the main script directly:

```bash
python process_audio_pipeline.py
```

### Advanced Usage

```bash
# Process specific directory
python run_processing.py --data-dir data/04_Carnatic_varnam

# Resume from checkpoint (if interrupted)
python run_processing.py --resume

# Custom chunk duration (20 seconds)
python run_processing.py --chunk-duration 20

# More aggressive filtering (remove bottom 20% by energy)
python run_processing.py --energy-threshold 20

# Dry run to see what would be processed
python run_processing.py --dry-run
```

## Output Structure

```
processed/
├── processed_audio/           # Processed 30-second chunks (WAV)
├── separated/
│   ├── drums/                # Mridangam/Tabla/Percussion tracks (always)
│   ├── accompaniment/        # Violin/Tanpura/Other instruments (always)
│   └── vocals/               # Vocal tracks (if detected)
└── metadata.csv              # Detailed metadata
```

## CSV Metadata

The generated `metadata.csv` includes:

- `datapoint` - Path to processed audio chunk
- `original_file` - Relative path to original source file
- **`original_file_path`** - Absolute path to original source file
- `chunk_index` - Chunk number within original file
- `duration` - Duration in seconds
- `sample_rate` - Audio sample rate
- **`raga`** - Raga name (automatically extracted from dataset annotations)
- **`thaat`** - Thaat/parent scale (for Hindustani music, if applicable)
- **`dataset_source`** - Source dataset (01-08)
- **`drums_path`** - Path to drums/mridangam/percussion stem (always present)
- **`accompaniment_path`** - Path to violin/tanpura/accompaniment stem (always present)
- `vocals_path` - Path to vocals stem (if vocal detected, else empty)
- `vocal_instrumental` - "vocal" or "instrumental"
- `spectral_energy` - Energy metric for filtering
- `processing_timestamp` - Processing timestamp

### Raga Extraction

The pipeline automatically extracts raga information from each dataset's annotations:
- **Saraga (Carnatic/Hindustani)**: From JSON metadata files
- **Indian-Music-Raga**: From filename patterns
- **Carnatic Varnam**: From directory structure and filenames
- **ThaatRagaForest**: Both thaat and raga from hierarchical directories
- **MelodicSimilarityDataset**: From directory names
- **Raga Ornamentation**: From filenames and CSV metadata
- **Mridangam Tani**: Marked as "not_applicable" (percussion only)

### Source Separation for Indian Classical Music

The pipeline separates each audio chunk into:
- **Drums** (Mridangam/Tabla/Percussion) - Always extracted, saved separately
- **Accompaniment** (Violin/Tanpura/Harmonium/Other) - Always extracted, includes all melodic accompaniment
- **Vocals** - Extracted if vocals detected

For **instrumental** clips, drums are still separated from accompaniment, giving you clean percussion and accompaniment tracks.

## Documentation

- **[RAGA_EXTRACTION_GUIDE.md](RAGA_EXTRACTION_GUIDE.md)** - Complete guide to raga extraction
- **[audio_config.py](audio_config.py)** - Configuration options
- **[example_usage.py](example_usage.py)** - Basic usage examples
- **[example_raga_usage.py](example_raga_usage.py)** - Raga-specific examples
- **[test_raga_extraction.py](test_raga_extraction.py)** - Test raga extraction

## Configuration

Edit `CONFIG` in `process_audio_pipeline.py` or use command-line arguments:

```python
CONFIG = {
    'target_loudness': -18.0,        # LUFS
    'chunk_duration': 30.0,          # seconds
    'sample_rate': 22050,            # Hz
    'energy_threshold_percentile': 10,  # Filter threshold
    'silence_threshold_db': 40,      # Silence trimming
    'min_chunk_duration': 5.0,       # Minimum chunk length
}
```

## Requirements

- Python 3.9+
- librosa
- soundfile
- pyloudnorm
- demucs
- pandas
- numpy
- tqdm
- torch
- torchaudio

## GPU Acceleration

For faster processing, install PyTorch with CUDA support:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Dataset Structure

The pipeline expects audio files in the `data/` directory:

```
data/
├── 01_saraga_carnatic/
├── 02_Indian-Music-Raga/
├── 03_Saraga_hindustani/
├── 04_Carnatic_varnam/
└── ...
```

Supported formats: `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`

## Examples

### Process Single File

```python
from process_audio_pipeline import process_audio_file
from pathlib import Path

metadata = process_audio_file(Path("data/audio.mp3"))
```

### Load Data for Training

```python
import pandas as pd
import librosa

# Load metadata
df = pd.read_csv('processed/metadata.csv')

# Filter by raga
yaman_clips = df[df['raga'] == 'Yaman']

# Load audio
for idx, row in yaman_clips.iterrows():
    audio, sr = librosa.load(row['datapoint'], sr=22050)
    percussion, sr = librosa.load(row['percussion'], sr=22050)
    print(f"Processing {row['raga']} clip from {row['dataset_source']}")
    # Your training code here...
```

### Test Raga Extraction

```bash
# Test raga extraction on all datasets
python test_raga_extraction.py
```

## Checkpoint & Resume

The pipeline automatically saves progress every 5 files to prevent data loss:

- **Checkpoint file**: `processed/processing_checkpoint.json`
- **Auto-save**: Every 5 files and after any errors
- **Resume command**: `python run_processing.py --resume`
- **Clean-up**: Checkpoint file removed when processing completes

If processing is interrupted (Ctrl+C, power loss, error), simply run with `--resume`:

```bash
# Start processing
python run_processing.py --data-dir data/02_Indian-Music-Raga

# If interrupted, resume from checkpoint
python run_processing.py --data-dir data/02_Indian-Music-Raga --resume
```

The pipeline will:
- Load existing progress
- Skip already processed files
- Continue from where it left off
- Append to existing metadata

## Performance

- Processing time depends on audio duration and hardware
- GPU acceleration recommended for Demucs
- Expect ~5-6x disk space increase (processed + separated stems)
- Progress saved every 5 files for safe interruption/resume

## Citation

If you use this pipeline, please cite:

```bibtex
@article{musicldm2023,
  title={MusicLDM: Enhancing Novelty in Text-to-Music Generation Using Beat-Synchronous Mixup Strategies},
  author={Chen, Ke and Wu, Yusong and Liu, Haohe},
  year={2023}
}
```

## License

See [MusicLDM/LICENSE](MusicLDM/LICENSE) for details.

## Troubleshooting

### Demucs not found
```bash
pip install demucs
```

### Out of memory
- Reduce `chunk_duration`
- Use GPU for processing
- Process files in smaller batches

### No audio files found
- Check `data/` directory exists
- Verify file extensions are supported
- Check files aren't in `.gitignore`

## Contact

For issues or questions, please open an issue on GitHub.
