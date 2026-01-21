"""
Audio Processing Pipeline for Raga-LDM Dataset

This script processes audio files through multiple stages:
1. Mono conversion
2. Loudness normalization (-18 LUFS)
3. Silence trimming
4. Chunking to 30-second segments
5. Low-energy/drone filtering
6. Source separation using Demucs
7. Metadata CSV generation
"""

import os
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import json
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import pyloudnorm as pyln
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'target_loudness': -18.0,  # LUFS
    'chunk_duration': 30.0,  # seconds
    'sample_rate': 22050,  # Hz
    'energy_threshold_percentile': 10,  # Remove bottom 10% by energy
    'silence_threshold_db': 40,  # dB below peak for silence trimming
    'min_chunk_duration': 5.0,  # Minimum chunk duration to keep
    'audio_extensions': ['.mp3', '.wav', '.flac', '.ogg', '.m4a'],
    'output_dir': 'processed',
    'data_dir': 'data',
    'num_workers': None,  # None = use all CPU cores
    'gpu_batch_size': 8,  # Batch size for GPU source separation
    'max_workers_files': None,  # Max workers for file processing (None = auto)
    'two_pass_separation': True,  # Use two-pass ensemble for better drum separation
    'drum_combination_weight': 0.7,  # Weight for first pass drums (0.7) vs second pass (0.3)
    'use_model_ensemble': True,  # Use ensemble of Demucs + MDX-Net for better separation
    'mdx_model': 'mdx_extra',  # MDX model to use ('mdx_extra', 'mdx', etc.)
    'fusion_method': 'weighted_transient',  # 'max_tf', 'weighted_transient', or 'average'
    'mdx_weight': 0.4,  # Weight for MDX drums in fusion (0.4) vs Demucs (0.6)
    'separation_shifts': 5,  # Number of shifts for equivariant stabilization (1-10, higher=better quality but slower)
    'separation_overlap': 0.25,  # Overlap between chunks (0.0-1.0)
    'use_wiener_filtering': True,  # Apply Wiener filtering for residual suppression
    'use_spectral_gating': True,  # Apply spectral gating to remove artifacts
    'use_cross_stem_consistency': True,  # Ensure stems sum to original (energy balance)
    'freq_dependent_fusion': True,  # Use frequency-dependent fusion weights
    'artifact_threshold_db': -40,  # Threshold for artifact detection in dB
}


def ensure_directories():
    """Create necessary output directories."""
    dirs = [
        Path(CONFIG['output_dir']),
        Path(CONFIG['output_dir']) / 'processed_audio',
        Path(CONFIG['output_dir']) / 'separated' / 'drums',
        Path(CONFIG['output_dir']) / 'separated' / 'accompaniment',
        Path(CONFIG['output_dir']) / 'separated' / 'vocals',
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directories created in: {CONFIG['output_dir']}")


def get_checkpoint_path():
    """Get path to checkpoint file."""
    return Path(CONFIG['output_dir']) / 'processing_checkpoint.json'


def load_checkpoint() -> Dict:
    """
    Load processing checkpoint.
    
    Returns:
        Dictionary with processed files and metadata
    """
    checkpoint_path = get_checkpoint_path()
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            return {'processed_files': [], 'metadata': []}
    return {'processed_files': [], 'metadata': []}


def save_checkpoint(processed_files: List[str], metadata: List[Dict]):
    """
    Save processing checkpoint.
    
    Args:
        processed_files: List of processed file paths
        metadata: List of metadata dictionaries
    """
    checkpoint_path = get_checkpoint_path()
    try:
        checkpoint_data = {
            'processed_files': processed_files,
            'metadata': metadata,
            'last_updated': datetime.now().isoformat(),
            'total_processed': len(processed_files)
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        logger.info(f"Checkpoint saved: {len(processed_files)} files processed")
    except Exception as e:
        logger.warning(f"Could not save checkpoint: {e}")


def is_file_already_processed(file_path: Path, processed_files: List[str]) -> bool:
    """
    Check if a file has already been processed.
    
    Args:
        file_path: Path to audio file
        processed_files: List of already processed file paths
        
    Returns:
        True if file was already processed
    """
    return str(file_path) in processed_files


def find_audio_files(data_dir: str) -> List[Path]:
    """
    Recursively find all audio files in the data directory.
    
    Args:
        data_dir: Root directory to search
        
    Returns:
        List of Path objects for all audio files found
    """
    audio_files = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return []
    
    for ext in CONFIG['audio_extensions']:
        audio_files.extend(data_path.rglob(f'*{ext}'))
    
    logger.info(f"Found {len(audio_files)} audio files in {data_dir}")
    return sorted(audio_files)


# ============================================================================
# Raga Extraction Functions
# ============================================================================

def extract_raga_saraga_carnatic(file_path: Path) -> str:
    """
    Extract raga from Saraga Carnatic JSON metadata.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Raga name or 'unknown'
    """
    try:
        json_path = file_path.with_suffix('.json')
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                raaga = data.get('raaga', [])
                if raaga and len(raaga) > 0:
                    if isinstance(raaga[0], dict):
                        # Try both 'name' and 'common_name'
                        raga_name = raaga[0].get('name') or raaga[0].get('common_name', 'unknown')
                        return raga_name
                    return str(raaga[0])
        return 'unknown'
    except Exception as e:
        logger.debug(f"Error extracting raga from Saraga Carnatic: {e}")
        return 'unknown'


def extract_raga_indian_music(file_path: Path) -> str:
    """
    Extract raga from filename pattern (ragaName##.wav).
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Raga name or 'unknown'
    """
    try:
        filename = file_path.stem
        # Remove trailing digits
        raga = re.sub(r'\d+$', '', filename)
        if raga and raga != filename:
            return raga.capitalize()
        return 'unknown'
    except Exception as e:
        logger.debug(f"Error extracting raga from Indian Music filename: {e}")
        return 'unknown'


def extract_raga_saraga_hindustani(file_path: Path) -> str:
    """
    Extract raga from Saraga Hindustani JSON metadata.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Raga name or 'unknown'
    """
    try:
        json_path = file_path.with_suffix('.json')
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                raags = data.get('raags', [])
                if raags and len(raags) > 0:
                    if isinstance(raags[0], dict):
                        # Use 'common_name' field as specified
                        return raags[0].get('common_name', 'unknown')
                    return str(raags[0])
        return 'unknown'
    except Exception as e:
        logger.debug(f"Error extracting raga from Saraga Hindustani: {e}")
        return 'unknown'


def extract_raga_carnatic_varnam(file_path: Path) -> str:
    """
    Extract raga from Carnatic Varnam directory structure.
    
    Raga names are folders in notations_annotations/notations, find them in filename
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Raga name or 'unknown'
    """
    try:
        filename = file_path.stem.lower()
        
        # Look for notations directory
        parent_dir = file_path.parent.parent
        notations_dir = parent_dir / 'Notations_Annotations' / 'notations'
        
        if notations_dir.exists():
            # List raga directories
            raga_dirs = [d.name for d in notations_dir.iterdir() if d.is_dir()]
            # Try to match with filename
            for raga in raga_dirs:
                if raga.lower() in filename:
                    return raga.capitalize()
        
        return 'unknown'
    except Exception as e:
        logger.debug(f"Error extracting raga from Carnatic Varnam: {e}")
        return 'unknown'


def extract_raga_thaat_forest(file_path: Path) -> Dict[str, str]:
    """
    Extract both thaat and raga from ThaatRagaForest directory structure.
    
    Raga is the name of the folder where audio files appear.
    Pattern: Thaat (thaat)/Raga/file.mp3
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dict with 'raga' and 'thaat' keys
    """
    try:
        # Raga is the immediate parent directory
        raga = file_path.parent.name
        
        parts = file_path.parts
        thaat = ''
        
        # Find the thaat from directory structure
        for i, part in enumerate(parts):
            if '(thaat)' in part.lower():
                # Extract thaat name (remove "(thaat)" suffix)
                thaat = part.replace(' (thaat)', '').replace(' (Thaat)', '')
                break
        
        return {'raga': raga, 'thaat': thaat}
    except Exception as e:
        logger.debug(f"Error extracting raga from ThaatRagaForest: {e}")
        return {'raga': 'unknown', 'thaat': 'unknown'}


def extract_raga_melodic_similarity(file_path: Path) -> str:
    """
    Extract raga from MelodicSimilarityDataset directory name.
    
    For Carnatic: ragas are sahana, kamboji, kalyani, bhairavi, varali
    For Hindustani: no raga extraction
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Raga name or 'unknown'
    """
    try:
        # Check if Carnatic or Hindustani
        path_str = str(file_path).lower()
        
        if 'hindustani' in path_str:
            return 'unknown'  # Hindustani doesn't have raga info
        
        # For Carnatic, find specific ragas in folder name
        carnatic_ragas = ['sahana', 'kamboji', 'kalyani', 'bhairavi', 'varali']
        dir_name = file_path.parent.name.lower()
        
        for raga in carnatic_ragas:
            if raga in dir_name:
                return raga.capitalize()
        
        return 'unknown'
    except Exception as e:
        logger.debug(f"Error extracting raga from Melodic Similarity: {e}")
        return 'unknown'


def extract_raga_ornamentation(file_path: Path) -> str:
    """
    Extract raga from Raga Ornamentation dataset.
    
    Only process audio in ROD/audio directory.
    Ragas in filenames: bageshree, bhoopali, bhairav
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Raga name or 'unknown'
    """
    try:
        # Only process ROD/audio files
        if 'ROD' not in file_path.parts or 'audio' not in file_path.parts:
            return 'unknown'
        
        filename = file_path.stem.lower()
        
        # Known ragas in this dataset
        rod_ragas = ['bageshree', 'bhoopali', 'bhairav', 'darbari']
        
        # Extract raga from filename (usually after the number prefix)
        # Pattern: 002_ragaName_description.wav
        for raga in rod_ragas:
            if raga in filename:
                return raga.capitalize()
        
        return 'unknown'
    except Exception as e:
        logger.debug(f"Error extracting raga from Ornamentation: {e}")
        return 'unknown'


def determine_dataset_source(file_path: Path, data_dir: str) -> str:
    """
    Determine which dataset the file belongs to based on path.
    
    Args:
        file_path: Path to audio file
        data_dir: Root data directory
        
    Returns:
        Dataset identifier (01-08) or 'unknown'
    """
    try:
        path_str = str(file_path)
        
        if '01_saraga_carnatic' in path_str or 'saraga' in path_str.lower() and 'carnatic' in path_str.lower():
            return '01_saraga_carnatic'
        elif '02_Indian-Music-Raga' in path_str or 'indian-music-raga' in path_str.lower():
            return '02_Indian-Music-Raga'
        elif '03_Saraga_hindustani' in path_str or 'saraga' in path_str.lower() and 'hindustani' in path_str.lower():
            return '03_Saraga_hindustani'
        elif '04_Carnatic_varnam' in path_str or 'carnatic_varnam' in path_str.lower():
            return '04_Carnatic_varnam'
        elif '05_Mridangam_Tani' in path_str or 'mridangam' in path_str.lower():
            return '05_Mridangam_Tani'
        elif '06_thaatRagaForest' in path_str or 'thaat' in path_str.lower():
            return '06_thaatRagaForest'
        elif '07_MelodicSimilarityDataset' in path_str or 'MelodicSimilarity' in path_str:
            return '07_MelodicSimilarityDataset'
        elif '08_raga_ornamentation_dataset' in path_str or 'ornamentation' in path_str.lower():
            return '08_raga_ornamentation_dataset'
        else:
            return 'unknown'
    except Exception as e:
        logger.debug(f"Error determining dataset source: {e}")
        return 'unknown'


def extract_raga_from_path(file_path: Path, data_dir: str = 'data') -> Dict[str, str]:
    """
    Extract raga information based on dataset structure.
    
    Args:
        file_path: Path to audio file
        data_dir: Root data directory
        
    Returns:
        Dictionary with 'raga', 'thaat', and 'dataset_source' keys
    """
    dataset = determine_dataset_source(file_path, data_dir)
    result = {
        'raga': 'unknown',
        'thaat': '',
        'dataset_source': dataset
    }
    
    try:
        if dataset == '01_saraga_carnatic':
            result['raga'] = extract_raga_saraga_carnatic(file_path)
        
        elif dataset == '02_Indian-Music-Raga':
            result['raga'] = extract_raga_indian_music(file_path)
        
        elif dataset == '03_Saraga_hindustani':
            result['raga'] = extract_raga_saraga_hindustani(file_path)
        
        elif dataset == '04_Carnatic_varnam':
            result['raga'] = extract_raga_carnatic_varnam(file_path)
        
        elif dataset == '05_Mridangam_Tani':
            result['raga'] = 'not_applicable'  # Percussion only
        
        elif dataset == '06_thaatRagaForest':
            thaat_raga = extract_raga_thaat_forest(file_path)
            result['raga'] = thaat_raga['raga']
            result['thaat'] = thaat_raga['thaat']
        
        elif dataset == '07_MelodicSimilarityDataset':
            result['raga'] = extract_raga_melodic_similarity(file_path)
        
        elif dataset == '08_raga_ornamentation_dataset':
            result['raga'] = extract_raga_ornamentation(file_path)
        
        logger.debug(f"Extracted raga info for {file_path.name}: {result}")
        
    except Exception as e:
        logger.warning(f"Error extracting raga from {file_path}: {e}")
    
    return result


# ============================================================================
# Audio Processing Functions
# ============================================================================

def load_and_convert_to_mono(file_path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    """
    Load audio file and convert to mono.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        Tuple of (mono_audio, sample_rate)
    """
    try:
        # Load audio with librosa (automatically converts to mono if stereo)
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, sr
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise


def normalize_loudness(audio: np.ndarray, sr: int, target_loudness: float) -> np.ndarray:
    """
    Normalize audio to target loudness using pyloudnorm.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        target_loudness: Target loudness in LUFS
        
    Returns:
        Loudness-normalized audio
    """
    try:
        # Create a meter with the audio's sample rate
        meter = pyln.Meter(sr)
        
        # Measure the loudness
        loudness = meter.integrated_loudness(audio)
        
        # Check if loudness is valid
        if np.isnan(loudness) or np.isinf(loudness):
            logger.warning("Invalid loudness measurement, skipping normalization")
            return audio
        
        # Normalize audio to target loudness
        normalized_audio = pyln.normalize.loudness(audio, loudness, target_loudness)
        
        # Clip to prevent clipping artifacts
        normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
        
        return normalized_audio
    except Exception as e:
        logger.warning(f"Error normalizing loudness: {e}, returning original audio")
        return audio


def trim_silence(audio: np.ndarray, sr: int, threshold_db: float) -> np.ndarray:
    """
    Trim silence from beginning and end of audio.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        threshold_db: Threshold in dB below peak
        
    Returns:
        Trimmed audio
    """
    try:
        # Use librosa's trim function
        trimmed_audio, _ = librosa.effects.trim(
            audio, 
            top_db=threshold_db,
            frame_length=2048,
            hop_length=512
        )
        return trimmed_audio
    except Exception as e:
        logger.warning(f"Error trimming silence: {e}, returning original audio")
        return audio


def chunk_audio(audio: np.ndarray, sr: int, chunk_duration: float) -> List[np.ndarray]:
    """
    Split audio into fixed-duration chunks.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        chunk_duration: Duration of each chunk in seconds
        
    Returns:
        List of audio chunks
    """
    chunk_samples = int(chunk_duration * sr)
    chunks = []
    
    # Split audio into chunks
    for start in range(0, len(audio), chunk_samples):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        
        # Only keep chunks that meet minimum duration
        min_samples = int(CONFIG['min_chunk_duration'] * sr)
        if len(chunk) >= min_samples:
            # Pad last chunk if needed
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
            chunks.append(chunk)
    
    return chunks


def calculate_spectral_energy(audio: np.ndarray, sr: int) -> float:
    """
    Calculate spectral energy of audio signal.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        
    Returns:
        Spectral energy value
    """
    try:
        # Compute spectrogram
        spec = np.abs(librosa.stft(audio))
        
        # Calculate spectral energy
        energy = np.sum(spec ** 2)
        
        # Calculate spectral centroid variance (measure of spectral variation)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        centroid_variance = np.var(spectral_centroid)
        
        # Combine energy and variation for better drone detection
        combined_metric = energy * (1 + centroid_variance)
        
        return float(combined_metric)
    except Exception as e:
        logger.warning(f"Error calculating spectral energy: {e}")
        return 0.0


def filter_low_energy_chunks(chunks: List[np.ndarray], sr: int) -> Tuple[List[np.ndarray], List[float]]:
    """
    Filter out low-energy/drone-only chunks.
    
    Args:
        chunks: List of audio chunks
        sr: Sample rate
        
    Returns:
        Tuple of (filtered chunks, energy values)
    """
    # Calculate energy for all chunks in parallel
    num_workers = CONFIG.get('num_workers', mp.cpu_count())
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    with ThreadPoolExecutor(max_workers=min(num_workers, len(chunks))) as executor:
        energies = list(executor.map(
            lambda chunk: calculate_spectral_energy(chunk, sr),
            chunks
        ))
    
    if not energies:
        return [], []
    
    # Calculate threshold based on percentile
    threshold = np.percentile(energies, CONFIG['energy_threshold_percentile'])
    
    # Filter chunks
    filtered_chunks = []
    filtered_energies = []
    
    for chunk, energy in zip(chunks, energies):
        if energy > threshold:
            filtered_chunks.append(chunk)
            filtered_energies.append(energy)
    
    removed_count = len(chunks) - len(filtered_chunks)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} low-energy chunks (threshold: {threshold:.2e})")
    
    return filtered_chunks, filtered_energies


def process_audio_file(file_path: Path) -> List[Dict]:
    """
    Process a single audio file through all stages.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        List of metadata dictionaries for each processed chunk
    """
    logger.info(f"Processing: {file_path}")
    
    try:
        # Stage 1: Load and convert to mono
        audio, sr = load_and_convert_to_mono(file_path, CONFIG['sample_rate'])
        
        # Stage 2: Normalize loudness
        audio = normalize_loudness(audio, sr, CONFIG['target_loudness'])
        
        # Stage 3: Trim silence
        audio = trim_silence(audio, sr, CONFIG['silence_threshold_db'])
        
        # Stage 4: Chunk audio
        chunks = chunk_audio(audio, sr, CONFIG['chunk_duration'])
        
        if not chunks:
            logger.warning(f"No chunks created from {file_path}")
            return []
        
        # Stage 5: Filter low-energy chunks
        filtered_chunks, energies = filter_low_energy_chunks(chunks, sr)
        
        if not filtered_chunks:
            logger.warning(f"All chunks filtered out from {file_path}")
            return []
        
        # Extract raga information from source file
        raga_info = extract_raga_from_path(file_path, CONFIG['data_dir'])
        
        # Save processed chunks and collect metadata
        metadata_list = []
        try:
            relative_path = file_path.relative_to(Path(CONFIG['data_dir']))
        except ValueError:
            relative_path = file_path
        absolute_path = file_path.resolve()
        
        for idx, chunk in enumerate(filtered_chunks):
            # Generate output filename
            stem = file_path.stem
            output_filename = f"{stem}_chunk_{idx:03d}.wav"
            output_path = Path(CONFIG['output_dir']) / 'processed_audio' / output_filename
            
            # Save processed chunk
            sf.write(output_path, chunk, sr)
            
            # Create metadata entry
            metadata = {
                'datapoint': str(output_path),
                'original_file': str(relative_path),
                'original_file_path': str(absolute_path),
                'chunk_index': idx,
                'duration': len(chunk) / sr,
                'sample_rate': sr,
                'spectral_energy': energies[idx] if idx < len(energies) else 0.0,
                'raga': raga_info['raga'],
                'thaat': raga_info['thaat'],
                'dataset_source': raga_info['dataset_source'],
                'processing_timestamp': datetime.now().isoformat(),
            }
            
            metadata_list.append(metadata)
        
        logger.info(f"Processed {len(filtered_chunks)} chunks from {file_path.name}")
        return metadata_list
        
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return []


def detect_vocals(vocal_stem: np.ndarray, other_stems: np.ndarray, threshold_ratio: float = 0.1) -> bool:
    """
    Detect if vocals are present by comparing vocal stem energy to other stems.
    
    Args:
        vocal_stem: Vocal stem audio
        other_stems: Other stems combined audio
        threshold_ratio: Minimum ratio of vocal energy to be considered "vocal"
        
    Returns:
        True if vocals detected, False otherwise
    """
    try:
        vocal_energy = np.sum(vocal_stem ** 2)
        other_energy = np.sum(other_stems ** 2)
        
        if other_energy == 0:
            return False
        
        ratio = vocal_energy / other_energy
        return ratio > threshold_ratio
    except Exception as e:
        logger.warning(f"Error detecting vocals: {e}")
        return False


# Global model cache for batch processing
_global_demucs_model = None
_global_mdx_model = None
_global_device = None

def get_demucs_model(device='cuda'):
    """Get or create Demucs model (cached globally)."""
    global _global_demucs_model, _global_device
    if _global_demucs_model is None or _global_device != device:
        import torch
        from demucs.pretrained import get_model
        _global_demucs_model = get_model('htdemucs')
        _global_demucs_model.eval()
        if device == 'cuda' and torch.cuda.is_available():
            _global_demucs_model = _global_demucs_model.to(device)
            # Optimize for A100
            if hasattr(torch, 'compile') and torch.cuda.get_device_capability()[0] >= 8:
                try:
                    _global_demucs_model = torch.compile(_global_demucs_model, mode='reduce-overhead')
                    logger.info("Demucs model compiled with torch.compile for A100 optimization")
                except Exception as e:
                    logger.warning(f"Could not compile Demucs model: {e}")
        _global_device = device
    return _global_demucs_model


def get_mdx_model(model_name='mdx_extra', device='cuda'):
    """Get or create MDX-Net model (cached globally)."""
    global _global_mdx_model, _global_device
    if _global_mdx_model is None or _global_device != device:
        import torch
        from demucs.pretrained import get_model
        try:
            _global_mdx_model = get_model(model_name)
            _global_mdx_model.eval()
            if device == 'cuda' and torch.cuda.is_available():
                _global_mdx_model = _global_mdx_model.to(device)
                # Optimize for A100
                if hasattr(torch, 'compile') and torch.cuda.get_device_capability()[0] >= 8:
                    try:
                        _global_mdx_model = torch.compile(_global_mdx_model, mode='reduce-overhead')
                        logger.info("MDX model compiled with torch.compile for A100 optimization")
                    except Exception as e:
                        logger.warning(f"Could not compile MDX model: {e}")
            _global_device = device
        except Exception as e:
            logger.warning(f"Could not load MDX model {model_name}: {e}")
            return None
    return _global_mdx_model


def fuse_drums_tf_max(drums_demucs: np.ndarray, drums_mdx: np.ndarray, sr: int) -> np.ndarray:
    """
    Fuse drums using max operation in time-frequency domain.
    
    Args:
        drums_demucs: Drums from Demucs model
        drums_mdx: Drums from MDX model
        sr: Sample rate
        
    Returns:
        Fused drums
    """
    # Convert to mono if needed
    if drums_demucs.ndim > 1:
        drums_demucs = drums_demucs.mean(axis=0)
    if drums_mdx.ndim > 1:
        drums_mdx = drums_mdx.mean(axis=0)
    
    # Compute STFT for both
    stft_demucs = librosa.stft(drums_demucs, n_fft=2048, hop_length=512)
    stft_mdx = librosa.stft(drums_mdx, n_fft=2048, hop_length=512)
    
    # Take max magnitude in TF domain
    magnitude_demucs = np.abs(stft_demucs)
    magnitude_mdx = np.abs(stft_mdx)
    phase_demucs = np.angle(stft_demucs)
    phase_mdx = np.angle(stft_mdx)
    
    # Max magnitude, use phase from the one with higher magnitude
    magnitude_fused = np.maximum(magnitude_demucs, magnitude_mdx)
    phase_fused = np.where(magnitude_demucs >= magnitude_mdx, phase_demucs, phase_mdx)
    
    # Reconstruct
    stft_fused = magnitude_fused * np.exp(1j * phase_fused)
    drums_fused = librosa.istft(stft_fused, hop_length=512)
    
    return drums_fused


def fuse_drums_weighted_transient(drums_demucs: np.ndarray, drums_mdx: np.ndarray, 
                                   sr: int, mdx_weight: float = 0.4) -> np.ndarray:
    """
    Fuse drums using weighted merge: transients from MDX, sustain from Demucs.
    
    Args:
        drums_demucs: Drums from Demucs model
        drums_mdx: Drums from MDX model
        sr: Sample rate
        mdx_weight: Weight for MDX (default 0.4)
        
    Returns:
        Fused drums
    """
    # Convert to mono if needed
    if drums_demucs.ndim > 1:
        drums_demucs = drums_demucs.mean(axis=0)
    if drums_mdx.ndim > 1:
        drums_mdx = drums_mdx.mean(axis=0)
    
    # Ensure same length
    min_len = min(len(drums_demucs), len(drums_mdx))
    drums_demucs = drums_demucs[:min_len]
    drums_mdx = drums_mdx[:min_len]
    
    # Compute STFT for both
    stft_demucs = librosa.stft(drums_demucs, n_fft=2048, hop_length=512)
    stft_mdx = librosa.stft(drums_mdx, n_fft=2048, hop_length=512)
    
    magnitude_demucs = np.abs(stft_demucs)
    magnitude_mdx = np.abs(stft_mdx)
    phase_demucs = np.angle(stft_demucs)
    phase_mdx = np.angle(stft_mdx)
    
    # Detect transients (high frequency, high energy changes)
    # MDX is better at transients, Demucs at sustain
    magnitude_diff = np.abs(magnitude_mdx - magnitude_demucs)
    transient_mask = magnitude_diff > np.percentile(magnitude_diff, 70)  # Top 30% of differences
    
    # Weighted combination: more MDX for transients, more Demucs for sustain
    weight_mdx = np.where(transient_mask, mdx_weight + 0.3, mdx_weight)  # Boost MDX for transients
    weight_demucs = 1.0 - weight_mdx
    
    # Combine magnitudes
    magnitude_fused = weight_demucs * magnitude_demucs + weight_mdx * magnitude_mdx
    
    # Use phase from the dominant source
    phase_fused = np.where(magnitude_mdx >= magnitude_demucs, phase_mdx, phase_demucs)
    
    # Reconstruct
    stft_fused = magnitude_fused * np.exp(1j * phase_fused)
    drums_fused = librosa.istft(stft_fused, hop_length=512)
    
    return drums_fused


def fuse_drums_average(drums_demucs: np.ndarray, drums_mdx: np.ndarray, 
                       sr: int, mdx_weight: float = 0.4) -> np.ndarray:
    """
    Simple weighted average fusion.
    
    Args:
        drums_demucs: Drums from Demucs model
        drums_mdx: Drums from MDX model
        sr: Sample rate
        mdx_weight: Weight for MDX (default 0.4)
        
    Returns:
        Fused drums
    """
    # Convert to mono if needed
    if drums_demucs.ndim > 1:
        drums_demucs = drums_demucs.mean(axis=0)
    if drums_mdx.ndim > 1:
        drums_mdx = drums_mdx.mean(axis=0)
    
    # Ensure same length
    min_len = min(len(drums_demucs), len(drums_mdx))
    drums_demucs = drums_demucs[:min_len]
    drums_mdx = drums_mdx[:min_len]
    
    # Weighted average
    demucs_weight = 1.0 - mdx_weight
    drums_fused = demucs_weight * drums_demucs + mdx_weight * drums_mdx
    
    return drums_fused


def apply_wiener_filtering(stem: np.ndarray, original: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply Wiener filtering for residual suppression.
    Helps remove artifacts and improve separation quality.
    
    Args:
        stem: Separated stem
        original: Original audio
        sr: Sample rate
        
    Returns:
        Filtered stem
    """
    try:
        # Convert to mono if needed
        if stem.ndim > 1:
            stem_mono = stem.mean(axis=0)
        else:
            stem_mono = stem.copy()
        if original.ndim > 1:
            original_mono = original.mean(axis=0)
        else:
            original_mono = original.copy()
        
        # Ensure same length
        min_len = min(len(stem_mono), len(original_mono))
        stem_mono = stem_mono[:min_len]
        original_mono = original_mono[:min_len]
        
        # Compute STFT
        stft_stem = librosa.stft(stem_mono, n_fft=2048, hop_length=512)
        stft_original = librosa.stft(original_mono, n_fft=2048, hop_length=512)
        
        # Compute power spectral densities
        psd_stem = np.abs(stft_stem) ** 2
        psd_original = np.abs(stft_original) ** 2
        
        # Wiener filter: H = P_stem / (P_stem + P_noise)
        # Where P_noise = P_original - P_stem (residual)
        psd_noise = np.maximum(psd_original - psd_stem, 1e-10)  # Prevent division by zero
        wiener_gain = psd_stem / (psd_stem + 0.1 * psd_noise)  # 0.1 is regularization factor
        
        # Apply filter
        stft_filtered = stft_stem * wiener_gain
        
        # Reconstruct
        stem_filtered = librosa.istft(stft_filtered, hop_length=512)
        
        # Restore original shape
        if stem.ndim > 1:
            stem_filtered = np.stack([stem_filtered, stem_filtered])
            return stem_filtered[:stem.shape[-1]]
        return stem_filtered
    except Exception as e:
        logger.warning(f"Error in Wiener filtering: {e}, returning original stem")
        return stem


def apply_spectral_gating(stem: np.ndarray, sr: int, threshold_db: float = -40) -> np.ndarray:
    """
    Apply spectral gating to remove low-energy artifacts.
    
    Args:
        stem: Separated stem
        sr: Sample rate
        threshold_db: Threshold in dB below peak
        
    Returns:
        Gated stem
    """
    try:
        # Convert to mono if needed
        if stem.ndim > 1:
            stem_mono = stem.mean(axis=0)
            is_stereo = True
        else:
            stem_mono = stem.copy()
            is_stereo = False
        
        # Compute STFT
        stft = librosa.stft(stem_mono, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Compute threshold
        max_magnitude = np.max(magnitude)
        threshold = max_magnitude * (10 ** (threshold_db / 20))
        
        # Create mask: keep only frequencies above threshold
        mask = magnitude > threshold
        
        # Apply mask
        stft_gated = stft * mask
        
        # Reconstruct
        stem_gated = librosa.istft(stft_gated, hop_length=512)
        
        # Restore original shape
        if is_stereo:
            stem_gated = np.stack([stem_gated, stem_gated])
            return stem_gated[:stem.shape[-1]]
        return stem_gated
    except Exception as e:
        logger.warning(f"Error in spectral gating: {e}, returning original stem")
        return stem


def apply_frequency_dependent_fusion(drums_demucs: np.ndarray, drums_mdx: np.ndarray, 
                                     sr: int, mdx_weight: float = 0.4) -> np.ndarray:
    """
    Frequency-dependent fusion: different weights for different frequency bands.
    MDX is better at high frequencies (transients), Demucs at low/mid frequencies.
    
    Args:
        drums_demucs: Drums from Demucs model
        drums_mdx: Drums from MDX model
        sr: Sample rate
        mdx_weight: Base weight for MDX (default 0.4)
        
    Returns:
        Fused drums
    """
    # Convert to mono if needed
    if drums_demucs.ndim > 1:
        drums_demucs = drums_demucs.mean(axis=0)
    if drums_mdx.ndim > 1:
        drums_mdx = drums_mdx.mean(axis=0)
    
    # Ensure same length
    min_len = min(len(drums_demucs), len(drums_mdx))
    drums_demucs = drums_demucs[:min_len]
    drums_mdx = drums_mdx[:min_len]
    
    # Compute STFT
    stft_demucs = librosa.stft(drums_demucs, n_fft=2048, hop_length=512)
    stft_mdx = librosa.stft(drums_mdx, n_fft=2048, hop_length=512)
    
    magnitude_demucs = np.abs(stft_demucs)
    magnitude_mdx = np.abs(stft_mdx)
    phase_demucs = np.angle(stft_demucs)
    phase_mdx = np.angle(stft_mdx)
    
    # Frequency-dependent weights
    n_freq_bins = magnitude_demucs.shape[0]
    freq_weights = np.linspace(0, 1, n_freq_bins)  # 0 at low freq, 1 at high freq
    
    # Higher weight for MDX at high frequencies (transients)
    # Lower weight for MDX at low frequencies (sustain)
    mdx_weights = mdx_weight + 0.3 * freq_weights[:, np.newaxis]  # Boost high freq
    mdx_weights = np.clip(mdx_weights, 0.1, 0.8)  # Limit range
    demucs_weights = 1.0 - mdx_weights
    
    # Combine magnitudes
    magnitude_fused = demucs_weights * magnitude_demucs + mdx_weights * magnitude_mdx
    
    # Use phase from dominant source
    phase_fused = np.where(magnitude_mdx >= magnitude_demucs, phase_mdx, phase_demucs)
    
    # Reconstruct
    stft_fused = magnitude_fused * np.exp(1j * phase_fused)
    drums_fused = librosa.istft(stft_fused, hop_length=512)
    
    return drums_fused


def separate_sources_batch(audio_paths: List[Path]) -> List[Dict[str, Optional[Path]]]:
    """
    Perform ensemble source separation on a batch of audio files using GPU.
    
    Process:
    1. Ensemble separation: Run both Demucs (HTDemucs) and MDX-Net models in parallel
    2. Fusion: Combine drums using TF-domain fusion (max or weighted transient)
    3. Two-pass refinement: Re-separate accompaniment to extract missed drums
    4. Final combination: Merge all drum sources
    
    Args:
        audio_paths: List of paths to processed audio files
        
    Returns:
        List of dictionaries with paths to separated stems
    """
    try:
        import torch
        from demucs.apply import apply_model
        
        if torch.cuda.is_available():
            device = 'cuda'
            # Clear cache before processing
            torch.cuda.empty_cache()
            logger.debug(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            logger.warning("CUDA not available, using CPU for batch processing")
        
        # Get cached models
        demucs_model = get_demucs_model(device)
        use_ensemble = CONFIG.get('use_model_ensemble', True)
        mdx_model = None
        if use_ensemble:
            mdx_model_name = CONFIG.get('mdx_model', 'mdx_extra')
            mdx_model = get_mdx_model(mdx_model_name, device)
            if mdx_model is None:
                logger.warning("MDX model not available, falling back to Demucs only")
                use_ensemble = False
        
        results = []
        
        # Process in batches
        batch_size = CONFIG.get('gpu_batch_size', 8)
        for batch_start in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[batch_start:batch_start + batch_size]
            batch_audio = []
            batch_srs = []
            batch_original_lengths = []
            
            # Load all audio in batch
            for audio_path in batch_paths:
                audio, sr = librosa.load(audio_path, sr=44100, mono=False)
                if audio.ndim == 1:
                    audio = np.stack([audio, audio])
                original_length = audio.shape[-1]
                batch_audio.append(audio)
                batch_srs.append(sr)
                batch_original_lengths.append(original_length)
            
            # ========== FIRST PASS: Initial separation with Demucs ==========
            # Stack into batch tensor (pad to same length)
            max_length = max(a.shape[-1] for a in batch_audio)
            batch_tensor = []
            for audio in batch_audio:
                if audio.shape[-1] < max_length:
                    padding = max_length - audio.shape[-1]
                    audio = np.pad(audio, ((0, 0), (0, padding)), mode='constant')
                batch_tensor.append(audio)
            
            batch_tensor = np.stack(batch_tensor)
            audio_tensor = torch.from_numpy(batch_tensor).float()
            
            if device == 'cuda':
                audio_tensor = audio_tensor.to(device)
            
            # Get separation parameters from config
            shifts = CONFIG.get('separation_shifts', 5)
            overlap = CONFIG.get('separation_overlap', 0.25)
            
            # Apply Demucs model to batch
            with torch.no_grad():
                sources_demucs = apply_model(
                    demucs_model, 
                    audio_tensor, 
                    device=device, 
                    shifts=shifts, 
                    split=True, 
                    overlap=overlap
                )
            
            # Apply MDX model if ensemble is enabled
            sources_mdx = None
            if use_ensemble and mdx_model is not None:
                with torch.no_grad():
                    sources_mdx = apply_model(
                        mdx_model,
                        audio_tensor,
                        device=device,
                        shifts=shifts,
                        split=True,
                        overlap=overlap
                    )
                logger.debug(f"Applied MDX model for ensemble separation on batch")
            
            # Clear GPU cache after first pass
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            # Clear GPU cache after first pass
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            # ========== SECOND PASS: Re-separate accompaniment ==========
            # Prepare accompaniment stems for second pass
            use_two_pass = CONFIG.get('two_pass_separation', True)
            accompaniment_batch = []
            accompaniment_indices = []
            
            if use_two_pass:
                for idx, audio_path in enumerate(batch_paths):
                    original_length = batch_original_lengths[idx]
                    # Get Demucs stems for second pass preparation
                    stems_demucs_temp = {
                        'drums': sources_demucs[idx][0][:, :original_length].cpu().numpy(),
                        'bass': sources_demucs[idx][1][:, :original_length].cpu().numpy(),
                        'other': sources_demucs[idx][2][:, :original_length].cpu().numpy(),
                        'vocals': sources_demucs[idx][3][:, :original_length].cpu().numpy(),
                    }
                    
                    # Create initial accompaniment (bass + other) for second pass
                    # Note: This will be cleaned later after ensemble fusion
                    accompaniment = stems_demucs_temp['bass'] + stems_demucs_temp['other']
                    
                    # Only process if accompaniment has significant energy
                    accompaniment_energy = np.sum(accompaniment ** 2)
                    if accompaniment_energy > 1e-6:  # Threshold to avoid processing silence
                        accompaniment_batch.append(accompaniment)
                        accompaniment_indices.append(idx)
            
            # Process accompaniment in second pass if we have any
            drums_pass2_dict = {}
            if use_two_pass and accompaniment_batch:
                # Stack accompaniment for batch processing
                max_length_acc = max(a.shape[-1] for a in accompaniment_batch)
                acc_tensor_list = []
                for acc in accompaniment_batch:
                    if acc.shape[-1] < max_length_acc:
                        padding = max_length_acc - acc.shape[-1]
                        acc = np.pad(acc, ((0, 0), (0, padding)), mode='constant')
                    acc_tensor_list.append(acc)
                
                acc_tensor = np.stack(acc_tensor_list)
                acc_tensor_torch = torch.from_numpy(acc_tensor).float()
                
                if device == 'cuda':
                    acc_tensor_torch = acc_tensor_torch.to(device)
                
                # Apply model to accompaniment (second pass) - use Demucs model
                with torch.no_grad():
                    sources_pass2 = apply_model(
                        demucs_model,
                        acc_tensor_torch,
                        device=device,
                        shifts=1,
                        split=True,
                        overlap=0.25
                    )
                
                # Extract drums from second pass
                for batch_idx, orig_idx in enumerate(accompaniment_indices):
                    original_length = batch_original_lengths[orig_idx]
                    # Get drums from second pass (these are drums that were missed in first pass)
                    drums_pass2 = sources_pass2[batch_idx][0][:, :original_length].cpu().numpy()
                    drums_pass2_dict[orig_idx] = drums_pass2
                
                # Clear GPU cache after second pass
                if device == 'cuda':
                    torch.cuda.empty_cache()
            
            # ========== COMBINE RESULTS ==========
            # Process each item in batch
            for idx, audio_path in enumerate(batch_paths):
                original_length = batch_original_lengths[idx]
                sr = batch_srs[idx]
                
                # Get Demucs stems
                stems_demucs = {
                    'drums': sources_demucs[idx][0][:, :original_length].cpu().numpy(),
                    'bass': sources_demucs[idx][1][:, :original_length].cpu().numpy(),
                    'other': sources_demucs[idx][2][:, :original_length].cpu().numpy(),
                    'vocals': sources_demucs[idx][3][:, :original_length].cpu().numpy(),
                }
                
                # Get MDX stems if available
                drums_mdx = None
                if use_ensemble and sources_mdx is not None:
                    # MDX models typically output: drums, bass, other, vocals (same order)
                    drums_mdx = sources_mdx[idx][0][:, :original_length].cpu().numpy()
                
                # Start with original accompaniment
                accompaniment = stems_demucs['bass'] + stems_demucs['other']
                
                # ========== FUSE DRUMS FROM MULTIPLE SOURCES ==========
                drums_combined = stems_demucs['drums'].copy()
                
                # First: Combine with MDX if ensemble is enabled
                if drums_mdx is not None:
                    fusion_method = CONFIG.get('fusion_method', 'weighted_transient')
                    mdx_weight = CONFIG.get('mdx_weight', 0.4)
                    use_freq_dep = CONFIG.get('freq_dependent_fusion', True)
                    
                    if use_freq_dep and fusion_method != 'max_tf':
                        # Use frequency-dependent fusion for better quality
                        drums_combined = apply_frequency_dependent_fusion(
                            stems_demucs['drums'], drums_mdx, sr, mdx_weight
                        )
                        logger.debug(f"Fused drums using frequency-dependent fusion for {audio_path.stem}")
                    elif fusion_method == 'max_tf':
                        drums_combined = fuse_drums_tf_max(stems_demucs['drums'], drums_mdx, sr)
                    elif fusion_method == 'weighted_transient':
                        drums_combined = fuse_drums_weighted_transient(
                            stems_demucs['drums'], drums_mdx, sr, mdx_weight
                        )
                    else:  # 'average'
                        drums_combined = fuse_drums_average(
                            stems_demucs['drums'], drums_mdx, sr, mdx_weight
                        )
                    logger.debug(f"Fused drums from Demucs and MDX using {fusion_method} for {audio_path.stem}")
                    
                    # Remove fused drums from accompaniment to get cleaner accompaniment
                    # Convert drums_combined to same shape as accompaniment if needed
                    if drums_combined.ndim == 1 and accompaniment.ndim > 1:
                        drums_combined_stereo = np.stack([drums_combined, drums_combined])
                    elif drums_combined.ndim > 1 and accompaniment.ndim == 1:
                        drums_combined_stereo = drums_combined.mean(axis=0)
                    else:
                        drums_combined_stereo = drums_combined
                    
                    # Ensure same length
                    min_len = min(accompaniment.shape[-1], drums_combined_stereo.shape[-1])
                    accompaniment = accompaniment[..., :min_len]
                    drums_combined_stereo = drums_combined_stereo[..., :min_len]
                    
                    # Subtract fused drums from accompaniment
                    accompaniment = accompaniment - 0.5 * drums_combined_stereo  # Use 0.5 weight to avoid over-subtraction
                    # Ensure non-negative (prevent phase issues)
                    accompaniment = np.maximum(accompaniment, -np.abs(accompaniment) * 0.3)
                    logger.debug(f"Removed fused drums from accompaniment for {audio_path.stem}")
                
                # Second: Combine with second pass drums if two-pass is enabled
                weight_pass1 = CONFIG.get('drum_combination_weight', 0.7)
                weight_pass2 = 1.0 - weight_pass1
                
                if idx in drums_pass2_dict:
                    drums_pass2 = drums_pass2_dict[idx]
                    # Add drums from second pass (weighted combination)
                    drums_combined = weight_pass1 * drums_combined + weight_pass2 * drums_pass2
                    logger.debug(f"Combined drums from two passes for {audio_path.stem}")
                    
                    # Further clean accompaniment by removing second pass drums
                    if drums_pass2.ndim == 1 and accompaniment.ndim > 1:
                        drums_pass2_stereo = np.stack([drums_pass2, drums_pass2])
                    elif drums_pass2.ndim > 1 and accompaniment.ndim == 1:
                        drums_pass2_stereo = drums_pass2.mean(axis=0)
                    else:
                        drums_pass2_stereo = drums_pass2
                    
                    # Ensure same length
                    min_len = min(accompaniment.shape[-1], drums_pass2_stereo.shape[-1])
                    accompaniment = accompaniment[..., :min_len]
                    drums_pass2_stereo = drums_pass2_stereo[..., :min_len]
                    
                    # Subtract second pass drums from accompaniment
                    accompaniment = accompaniment - weight_pass2 * drums_pass2_stereo
                    # Ensure non-negative (prevent phase issues)
                    accompaniment = np.maximum(accompaniment, -np.abs(accompaniment) * 0.3)
                    logger.debug(f"Removed second pass drums from accompaniment for {audio_path.stem}")
                
                # Detect if vocals are present (use Demucs vocals)
                has_vocals = detect_vocals(
                    stems_demucs['vocals'].mean(axis=0) if stems_demucs['vocals'].ndim > 1 else stems_demucs['vocals'],
                    accompaniment.mean(axis=0) if accompaniment.ndim > 1 else accompaniment
                )
                
                # Save stems
                base_name = audio_path.stem
                output_paths = {}
                
                # Apply post-processing to drums
                use_wiener = CONFIG.get('use_wiener_filtering', True)
                use_gating = CONFIG.get('use_spectral_gating', True)
                artifact_threshold = CONFIG.get('artifact_threshold_db', -40)
                
                # Get original audio for Wiener filtering
                original_audio = batch_audio[idx][:, :original_length] if batch_audio[idx].ndim > 1 else batch_audio[idx][:original_length]
                
                if use_wiener:
                    drums_combined = apply_wiener_filtering(drums_combined, original_audio, sr)
                if use_gating:
                    drums_combined = apply_spectral_gating(drums_combined, sr, artifact_threshold)
                
                # Save combined drums
                drums_path = Path(CONFIG['output_dir']) / 'separated' / 'drums' / f"{base_name}_drums.wav"
                drums_mono = drums_combined.mean(axis=0) if drums_combined.ndim > 1 else drums_combined
                # Normalize to prevent clipping
                max_val = np.max(np.abs(drums_mono))
                if max_val > 0.95:
                    drums_mono = drums_mono * (0.95 / max_val)
                sf.write(drums_path, drums_mono, sr)
                output_paths['drums_path'] = drums_path
                
                # Apply post-processing to accompaniment
                if use_wiener:
                    accompaniment = apply_wiener_filtering(accompaniment, original_audio, sr)
                if use_gating:
                    accompaniment = apply_spectral_gating(accompaniment, sr, artifact_threshold)
                
                # Save cleaned accompaniment
                accompaniment_path = Path(CONFIG['output_dir']) / 'separated' / 'accompaniment' / f"{base_name}_accompaniment.wav"
                accompaniment_mono = accompaniment.mean(axis=0) if accompaniment.ndim > 1 else accompaniment
                # Normalize to prevent clipping
                max_val = np.max(np.abs(accompaniment_mono))
                if max_val > 0.95:
                    accompaniment_mono = accompaniment_mono * (0.95 / max_val)
                sf.write(accompaniment_path, accompaniment_mono, sr)
                output_paths['accompaniment_path'] = accompaniment_path
                
                if has_vocals:
                    vocals_path = Path(CONFIG['output_dir']) / 'separated' / 'vocals' / f"{base_name}_vocals.wav"
                    # Use Demucs vocals (they're usually better for vocals)
                    vocals_mono = stems_demucs['vocals'].mean(axis=0) if stems_demucs['vocals'].ndim > 1 else stems_demucs['vocals']
                    sf.write(vocals_path, vocals_mono, sr)
                    output_paths['vocals_path'] = vocals_path
                    output_paths['vocal_instrumental'] = 'vocal'
                else:
                    output_paths['vocals_path'] = None
                    output_paths['vocal_instrumental'] = 'instrumental'
                
                results.append(output_paths)
        
        return results
        
    except ImportError as e:
        logger.error(f"Demucs not properly installed: {e}")
        logger.error("Please install demucs: pip install demucs")
        return [{
            'drums_path': None,
            'accompaniment_path': None,
            'vocals_path': None,
            'vocal_instrumental': 'unknown'
        } for _ in audio_paths]
    except Exception as e:
        logger.error(f"Error in batch source separation: {e}")
        return [{
            'drums_path': None,
            'accompaniment_path': None,
            'vocals_path': None,
            'vocal_instrumental': 'unknown'
        } for _ in audio_paths]


def separate_sources(audio_path: Path) -> Dict[str, Optional[Path]]:
    """
    Perform ensemble source separation using Demucs + MDX-Net with two-pass refinement.
    
    Process:
    1. Ensemble separation: Run both Demucs (HTDemucs) and MDX-Net models
    2. Fusion: Combine drums using TF-domain fusion (max or weighted transient)
    3. Two-pass refinement: Re-separate accompaniment to extract missed drums
    4. Final combination: Merge all drum sources
    
    Always extracts:
    - Drums/Mridangam (percussion) - enhanced with ensemble + two-pass approach
    - Violin/Accompaniment (other instruments) - cleaned of drums
    - Vocals (if present) - from Demucs (better for vocals)
    
    Args:
        audio_path: Path to processed audio file
        
    Returns:
        Dictionary with paths to separated stems
    """
    try:
        import torch
        from demucs.apply import apply_model
        
        if torch.cuda.is_available():
            device = 'cuda'
            logger.debug(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            logger.debug("CUDA not available, using CPU")
        
        # Load the audio
        audio, sr = librosa.load(audio_path, sr=44100, mono=False)
        
        # Ensure stereo for demucs (duplicate mono to stereo if needed)
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        
        # Get cached models
        demucs_model = get_demucs_model(device)
        use_ensemble = CONFIG.get('use_model_ensemble', True)
        mdx_model = None
        if use_ensemble:
            mdx_model_name = CONFIG.get('mdx_model', 'mdx_extra')
            mdx_model = get_mdx_model(mdx_model_name, device)
            if mdx_model is None:
                logger.warning("MDX model not available, falling back to Demucs only")
                use_ensemble = False
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        if device == 'cuda':
            audio_tensor = audio_tensor.to(device)
        
        # Get separation parameters from config
        shifts = CONFIG.get('separation_shifts', 5)
        overlap = CONFIG.get('separation_overlap', 0.25)
        
        # ========== FIRST PASS: Initial separation with Demucs ==========
        with torch.no_grad():
            sources_demucs = apply_model(demucs_model, audio_tensor, device=device, shifts=shifts, split=True, overlap=overlap)[0]
        
        # Sources order: drums, bass, other, vocals
        stems_demucs = {
            'drums': sources_demucs[0].cpu().numpy(),
            'bass': sources_demucs[1].cpu().numpy(),
            'other': sources_demucs[2].cpu().numpy(),
            'vocals': sources_demucs[3].cpu().numpy(),
        }
        
        # Apply MDX model if ensemble is enabled
        drums_mdx = None
        if use_ensemble and mdx_model is not None:
            with torch.no_grad():
                sources_mdx = apply_model(mdx_model, audio_tensor, device=device, shifts=shifts, split=True, overlap=overlap)[0]
                drums_mdx = sources_mdx[0].cpu().numpy()
                logger.debug(f"Applied MDX model for ensemble separation")
        
        # Clear GPU cache after first pass
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # Start with original accompaniment
        accompaniment = stems_demucs['bass'] + stems_demucs['other']
        
        # ========== FUSE DRUMS FROM MULTIPLE MODELS ==========
        drums_combined = stems_demucs['drums'].copy()
        
        # First: Combine with MDX if ensemble is enabled
        if drums_mdx is not None:
            fusion_method = CONFIG.get('fusion_method', 'weighted_transient')
            mdx_weight = CONFIG.get('mdx_weight', 0.4)
            use_freq_dep = CONFIG.get('freq_dependent_fusion', True)
            
            if use_freq_dep and fusion_method != 'max_tf':
                # Use frequency-dependent fusion for better quality
                drums_combined = apply_frequency_dependent_fusion(
                    stems_demucs['drums'], drums_mdx, sr, mdx_weight
                )
                logger.debug(f"Fused drums using frequency-dependent fusion")
            elif fusion_method == 'max_tf':
                drums_combined = fuse_drums_tf_max(stems_demucs['drums'], drums_mdx, sr)
            elif fusion_method == 'weighted_transient':
                drums_combined = fuse_drums_weighted_transient(
                    stems_demucs['drums'], drums_mdx, sr, mdx_weight
                )
            else:  # 'average'
                drums_combined = fuse_drums_average(
                    stems_demucs['drums'], drums_mdx, sr, mdx_weight
                )
            logger.debug(f"Fused drums from Demucs and MDX using {fusion_method}")
            
            # Remove fused drums from accompaniment to get cleaner accompaniment
            # Convert drums_combined to same shape as accompaniment if needed
            if drums_combined.ndim == 1 and accompaniment.ndim > 1:
                drums_combined_stereo = np.stack([drums_combined, drums_combined])
            elif drums_combined.ndim > 1 and accompaniment.ndim == 1:
                drums_combined_stereo = drums_combined.mean(axis=0)
            else:
                drums_combined_stereo = drums_combined
            
            # Ensure same length
            min_len = min(accompaniment.shape[-1], drums_combined_stereo.shape[-1])
            accompaniment = accompaniment[..., :min_len]
            drums_combined_stereo = drums_combined_stereo[..., :min_len]
            
            # Subtract fused drums from accompaniment
            accompaniment = accompaniment - 0.5 * drums_combined_stereo  # Use 0.5 weight to avoid over-subtraction
            # Ensure non-negative (prevent phase issues)
            accompaniment = np.maximum(accompaniment, -np.abs(accompaniment) * 0.3)
            logger.debug(f"Removed fused drums from accompaniment")
        
        # ========== SECOND PASS: Re-separate accompaniment ==========
        drums_pass2 = None
        use_two_pass = CONFIG.get('two_pass_separation', True)
        weight_pass1 = CONFIG.get('drum_combination_weight', 0.7)
        weight_pass2 = 1.0 - weight_pass1
        
        if use_two_pass:
            # Check if accompaniment has significant energy
            accompaniment_energy = np.sum(accompaniment ** 2)
            if accompaniment_energy > 1e-6:
                # Prepare accompaniment for second pass (use Demucs model)
                acc_tensor = torch.from_numpy(accompaniment).float().unsqueeze(0)
                if device == 'cuda':
                    acc_tensor = acc_tensor.to(device)
                
                # Apply model to accompaniment (second pass)
                with torch.no_grad():
                    sources_pass2 = apply_model(
                        demucs_model, 
                        acc_tensor, 
                        device=device, 
                        shifts=shifts, 
                        split=True, 
                        overlap=overlap
                    )[0]
                
                # Extract drums from second pass
                drums_pass2 = sources_pass2[0].cpu().numpy()
                logger.debug(f"Two-pass separation: extracted additional drums from accompaniment for {audio_path.stem}")
        
        # Combine with second pass drums
        if drums_pass2 is not None:
            drums_combined = weight_pass1 * drums_combined + weight_pass2 * drums_pass2
            
            # Further clean accompaniment by removing second pass drums
            if drums_pass2.ndim == 1 and accompaniment.ndim > 1:
                drums_pass2_stereo = np.stack([drums_pass2, drums_pass2])
            elif drums_pass2.ndim > 1 and accompaniment.ndim == 1:
                drums_pass2_stereo = drums_pass2.mean(axis=0)
            else:
                drums_pass2_stereo = drums_pass2
            
            # Ensure same length
            min_len = min(accompaniment.shape[-1], drums_pass2_stereo.shape[-1])
            accompaniment = accompaniment[..., :min_len]
            drums_pass2_stereo = drums_pass2_stereo[..., :min_len]
            
            # Subtract second pass drums from accompaniment
            accompaniment = accompaniment - weight_pass2 * drums_pass2_stereo
            # Ensure non-negative (prevent phase issues)
            accompaniment = np.maximum(accompaniment, -np.abs(accompaniment) * 0.3)
            logger.debug(f"Removed second pass drums from accompaniment")
        
        # Detect if vocals are present (use Demucs vocals)
        has_vocals = detect_vocals(
            stems_demucs['vocals'].mean(axis=0) if stems_demucs['vocals'].ndim > 1 else stems_demucs['vocals'],
            accompaniment.mean(axis=0) if accompaniment.ndim > 1 else accompaniment
        )
        
        # Save stems
        base_name = audio_path.stem
        output_paths = {}
        
        # Apply post-processing to drums
        use_wiener = CONFIG.get('use_wiener_filtering', True)
        use_gating = CONFIG.get('use_spectral_gating', True)
        artifact_threshold = CONFIG.get('artifact_threshold_db', -40)
        
        if use_wiener:
            drums_combined = apply_wiener_filtering(drums_combined, audio, sr)
        if use_gating:
            drums_combined = apply_spectral_gating(drums_combined, sr, artifact_threshold)
        
        # Save combined drums
        drums_path = Path(CONFIG['output_dir']) / 'separated' / 'drums' / f"{base_name}_drums.wav"
        drums_mono = drums_combined.mean(axis=0) if drums_combined.ndim > 1 else drums_combined
        # Normalize to prevent clipping
        max_val = np.max(np.abs(drums_mono))
        if max_val > 0.95:
            drums_mono = drums_mono * (0.95 / max_val)
        sf.write(drums_path, drums_mono, sr)
        output_paths['drums_path'] = drums_path
        
        # Apply post-processing to accompaniment
        if use_wiener:
            accompaniment = apply_wiener_filtering(accompaniment, audio, sr)
        if use_gating:
            accompaniment = apply_spectral_gating(accompaniment, sr, artifact_threshold)
        
        # Save cleaned accompaniment
        accompaniment_path = Path(CONFIG['output_dir']) / 'separated' / 'accompaniment' / f"{base_name}_accompaniment.wav"
        accompaniment_mono = accompaniment.mean(axis=0) if accompaniment.ndim > 1 else accompaniment
        # Normalize to prevent clipping
        max_val = np.max(np.abs(accompaniment_mono))
        if max_val > 0.95:
            accompaniment_mono = accompaniment_mono * (0.95 / max_val)
        sf.write(accompaniment_path, accompaniment_mono, sr)
        output_paths['accompaniment_path'] = accompaniment_path
        
        if has_vocals:
            # Save vocals (use Demucs vocals - they're usually better)
            vocals_path = Path(CONFIG['output_dir']) / 'separated' / 'vocals' / f"{base_name}_vocals.wav"
            vocals_mono = stems_demucs['vocals'].mean(axis=0) if stems_demucs['vocals'].ndim > 1 else stems_demucs['vocals']
            sf.write(vocals_path, vocals_mono, sr)
            output_paths['vocals_path'] = vocals_path
            output_paths['vocal_instrumental'] = 'vocal'
        else:
            output_paths['vocals_path'] = None
            output_paths['vocal_instrumental'] = 'instrumental'
        
        return output_paths
        
    except ImportError as e:
        logger.error(f"Demucs not properly installed: {e}")
        logger.error("Please install demucs: pip install demucs")
        return {
            'percussion': None,
            'other_path': None,
            'vocals_path': None,
            'instrumental_path': None,
            'vocal_instrumental': 'unknown'
        }
    except Exception as e:
        logger.error(f"Error in source separation for {audio_path}: {e}")
        return {
            'percussion': None,
            'other_path': None,
            'vocals_path': None,
            'instrumental_path': None,
            'vocal_instrumental': 'unknown'
        }


def process_single_file_wrapper(args: Tuple[Path, Dict]) -> Tuple[List[Dict], str]:
    """
    Wrapper function for processing a single file (for multiprocessing).
    
    Args:
        args: Tuple of (file_path, config_dict)
        
    Returns:
        Tuple of (metadata_list, file_path_str)
    """
    file_path, config = args
    try:
        # Set config for this worker
        global CONFIG
        original_config = CONFIG.copy()
        CONFIG.update(config)
        
        file_metadata = process_audio_file(file_path)
        
        # Restore original config
        CONFIG = original_config
        
        return file_metadata, str(file_path)
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return [], str(file_path)


def process_all_files(data_dir: str, resume: bool = False) -> pd.DataFrame:
    """
    Process all audio files and generate metadata CSV with parallel processing.
    
    Args:
        data_dir: Root directory containing audio files
        resume: Whether to resume from checkpoint
        
    Returns:
        DataFrame with all metadata
    """
    # Load checkpoint if resuming
    checkpoint = {'processed_files': [], 'metadata': []}
    if resume:
        checkpoint = load_checkpoint()
        logger.info(f"Resuming from checkpoint: {len(checkpoint['processed_files'])} files already processed")
    
    # Find all audio files
    audio_files = find_audio_files(data_dir)
    
    if not audio_files:
        logger.error("No audio files found!")
        return pd.DataFrame()
    
    # Filter out already processed files
    if resume and checkpoint['processed_files']:
        original_count = len(audio_files)
        audio_files = [f for f in audio_files if not is_file_already_processed(f, checkpoint['processed_files'])]
        skipped = original_count - len(audio_files)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already processed files")
    
    if not audio_files and checkpoint['metadata']:
        logger.info("All files already processed! Loading existing metadata.")
        return pd.DataFrame(checkpoint['metadata'])
    
    # Start with existing metadata if resuming
    all_metadata = checkpoint['metadata'].copy() if resume else []
    processed_files = checkpoint['processed_files'].copy() if resume else []
    
    # Determine number of workers
    max_workers = CONFIG.get('max_workers_files', None)
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(audio_files))
    
    logger.info(f"Processing {len(audio_files)} files with {max_workers} parallel workers")
    
    # Process files in parallel
    file_metadata_map = {}
    # Prepare config dict for workers (only serializable values)
    config_dict = {
        'target_loudness': CONFIG['target_loudness'],
        'chunk_duration': CONFIG['chunk_duration'],
        'sample_rate': CONFIG['sample_rate'],
        'energy_threshold_percentile': CONFIG['energy_threshold_percentile'],
        'silence_threshold_db': CONFIG['silence_threshold_db'],
        'min_chunk_duration': CONFIG['min_chunk_duration'],
        'output_dir': CONFIG['output_dir'],
        'data_dir': CONFIG['data_dir'],
        'num_workers': CONFIG.get('num_workers', None),
    }
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks with config
        future_to_file = {
            executor.submit(process_single_file_wrapper, (file_path, config_dict)): file_path 
            for file_path in audio_files
        }
        
        # Collect results as they complete
        for future in tqdm(as_completed(future_to_file), total=len(audio_files), desc="Processing files"):
            file_path = future_to_file[future]
            try:
                file_metadata, file_path_str = future.result()
                file_metadata_map[file_path_str] = file_metadata
                processed_files.append(file_path_str)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                processed_files.append(str(file_path))
    
    # Collect all chunks for batch source separation
    all_chunks_metadata = []
    for file_path_str, file_metadata in file_metadata_map.items():
        all_chunks_metadata.extend(file_metadata)
    
    logger.info(f"Processing {len(all_chunks_metadata)} chunks through source separation")
    
    # Batch process source separation on GPU
    if all_chunks_metadata:
        # Group chunks into batches for GPU processing
        batch_size = CONFIG.get('gpu_batch_size', 8)
        chunk_paths = [Path(metadata['datapoint']) for metadata in all_chunks_metadata]
        
        # Process in batches
        for batch_start in tqdm(range(0, len(chunk_paths), batch_size), desc="Separating sources"):
            batch_paths = chunk_paths[batch_start:batch_start + batch_size]
            batch_metadata = all_chunks_metadata[batch_start:batch_start + batch_size]
            
            # Perform batch source separation
            separation_results = separate_sources_batch(batch_paths)
            
            # Merge separation results into metadata
            for metadata, separation_paths in zip(batch_metadata, separation_results):
                metadata.update(separation_paths)
                
                # Convert Path objects to strings for CSV
                for key, value in metadata.items():
                    if isinstance(value, Path):
                        metadata[key] = str(value)
                
                all_metadata.append(metadata)
            
            # Save checkpoint periodically
            if (batch_start // batch_size + 1) % 10 == 0:
                save_checkpoint(processed_files, all_metadata)
    
    # Final checkpoint save
    save_checkpoint(processed_files, all_metadata)
    
    # Create DataFrame
    df = pd.DataFrame(all_metadata)
    
    # Reorder columns for clarity
    column_order = [
        'datapoint',
        'original_file',
        'original_file_path',
        'chunk_index',
        'duration',
        'sample_rate',
        'raga',
        'thaat',
        'dataset_source',
        'drums_path',
        'accompaniment_path',
        'vocals_path',
        'vocal_instrumental',
        'spectral_energy',
        'processing_timestamp',
    ]
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    if not df.empty:
        df = df[column_order]
    
    return df


def main(resume: bool = False):
    """
    Main execution function.
    
    Args:
        resume: Whether to resume from checkpoint
    """
    logger.info("=" * 80)
    logger.info("Audio Processing Pipeline for Raga-LDM Dataset")
    if resume:
        logger.info("RESUME MODE: Continuing from checkpoint")
    logger.info("=" * 80)
    
    # Ensure output directories exist
    ensure_directories()
    
    # Process all files
    logger.info(f"Starting processing of files in: {CONFIG['data_dir']}")
    metadata_df = process_all_files(CONFIG['data_dir'], resume=resume)
    
    if metadata_df.empty:
        logger.error("No data processed!")
        return
    
    # Save metadata CSV
    output_csv = Path(CONFIG['output_dir']) / 'metadata.csv'
    metadata_df.to_csv(output_csv, index=False)
    logger.info(f"Metadata saved to: {output_csv}")
    
    # Print summary statistics
    logger.info("=" * 80)
    logger.info("Processing Summary:")
    logger.info(f"Total chunks processed: {len(metadata_df)}")
    logger.info(f"Vocal clips: {(metadata_df['vocal_instrumental'] == 'vocal').sum()}")
    logger.info(f"Instrumental clips: {(metadata_df['vocal_instrumental'] == 'instrumental').sum()}")
    logger.info(f"Total duration: {metadata_df['duration'].sum() / 3600:.2f} hours")
    logger.info("=" * 80)
    logger.info("Processing complete!")
    
    # Clean up checkpoint file
    checkpoint_path = get_checkpoint_path()
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint file cleaned up")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()
    main(resume=args.resume)
