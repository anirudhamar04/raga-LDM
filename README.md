# Raga-LDM: Audio Processing Pipeline

This repository contains an audio processing pipeline for preparing Carnatic and Hindustani classical music datasets for machine learning applications.

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
