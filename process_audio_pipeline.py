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
}


def ensure_directories():
    """Create necessary output directories."""
    dirs = [
        Path(CONFIG['output_dir']),
        Path(CONFIG['output_dir']) / 'processed_audio',
        Path(CONFIG['output_dir']) / 'separated' / 'drums',
        Path(CONFIG['output_dir']) / 'separated' / 'other',
        Path(CONFIG['output_dir']) / 'separated' / 'vocals',
        Path(CONFIG['output_dir']) / 'separated' / 'instrumental',
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
    # Calculate energy for all chunks
    energies = [calculate_spectral_energy(chunk, sr) for chunk in chunks]
    
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
        relative_path = file_path.relative_to(CONFIG['data_dir']) if file_path.is_relative_to(CONFIG['data_dir']) else file_path
        
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


def separate_sources(audio_path: Path) -> Dict[str, Optional[Path]]:
    """
    Perform source separation using Demucs.
    
    Args:
        audio_path: Path to processed audio file
        
    Returns:
        Dictionary with paths to separated stems
    """
    try:
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        
        # Load the audio
        audio, sr = librosa.load(audio_path, sr=44100, mono=False)
        
        # Ensure stereo for demucs (duplicate mono to stereo if needed)
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        
        # Load demucs model (htdemucs is the best model)
        model = get_model('htdemucs')
        model.eval()
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        
        # Apply model
        with torch.no_grad():
            sources = apply_model(model, audio_tensor, device='cpu', shifts=1, split=True, overlap=0.25)[0]
        
        # Sources order: drums, bass, other, vocals
        # For Indian classical music, we want: drums (percussion), other (violin/accompaniment), vocals
        stems = {
            'drums': sources[0].numpy(),  # Drums/percussion
            'bass': sources[1].numpy(),   # Bass
            'other': sources[2].numpy(),  # Other instruments
            'vocals': sources[3].numpy(), # Vocals
        }
        
        # Combine bass and other for "other" category
        combined_other = stems['bass'] + stems['other']
        
        # Detect if vocals are present
        has_vocals = detect_vocals(
            stems['vocals'].mean(axis=0) if stems['vocals'].ndim > 1 else stems['vocals'],
            combined_other.mean(axis=0) if combined_other.ndim > 1 else combined_other
        )
        
        # Save stems
        base_name = audio_path.stem
        output_paths = {}
        
        # Always save drums/percussion
        drums_path = Path(CONFIG['output_dir']) / 'separated' / 'drums' / f"{base_name}_drums.wav"
        # Convert to mono for drums
        drums_mono = stems['drums'].mean(axis=0) if stems['drums'].ndim > 1 else stems['drums']
        sf.write(drums_path, drums_mono, sr)
        output_paths['percussion'] = drums_path
        
        # Save other/accompaniment
        other_path = Path(CONFIG['output_dir']) / 'separated' / 'other' / f"{base_name}_other.wav"
        other_mono = combined_other.mean(axis=0) if combined_other.ndim > 1 else combined_other
        sf.write(other_path, other_mono, sr)
        output_paths['other_path'] = other_path
        
        if has_vocals:
            # Save vocals
            vocals_path = Path(CONFIG['output_dir']) / 'separated' / 'vocals' / f"{base_name}_vocals.wav"
            vocals_mono = stems['vocals'].mean(axis=0) if stems['vocals'].ndim > 1 else stems['vocals']
            sf.write(vocals_path, vocals_mono, sr)
            output_paths['vocals_path'] = vocals_path
            output_paths['vocal_instrumental'] = 'vocal'
            output_paths['instrumental_path'] = None
        else:
            # Save as instrumental (combine all non-vocal stems)
            instrumental = stems['drums'] + combined_other
            instrumental_path = Path(CONFIG['output_dir']) / 'separated' / 'instrumental' / f"{base_name}_instrumental.wav"
            instrumental_mono = instrumental.mean(axis=0) if instrumental.ndim > 1 else instrumental
            sf.write(instrumental_path, instrumental_mono, sr)
            output_paths['instrumental_path'] = instrumental_path
            output_paths['vocal_instrumental'] = 'instrumental'
            output_paths['vocals_path'] = None
        
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


def process_all_files(data_dir: str, resume: bool = False) -> pd.DataFrame:
    """
    Process all audio files and generate metadata CSV.
    
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
    
    # Process remaining files
    for file_idx, file_path in enumerate(tqdm(audio_files, desc="Processing audio files")):
        try:
            # Stage 1-5: Basic processing and chunking
            file_metadata = process_audio_file(file_path)
            
            # Stage 6: Source separation for each chunk
            for metadata in tqdm(file_metadata, desc=f"Separating sources for {file_path.name}", leave=False):
                audio_path = Path(metadata['datapoint'])
                
                # Perform source separation
                separation_paths = separate_sources(audio_path)
                
                # Merge separation results into metadata
                metadata.update(separation_paths)
                
                # Convert Path objects to strings for CSV
                for key, value in metadata.items():
                    if isinstance(value, Path):
                        metadata[key] = str(value)
                
                all_metadata.append(metadata)
            
            # Mark file as processed
            processed_files.append(str(file_path))
            
            # Save checkpoint every 5 files
            if (file_idx + 1) % 5 == 0:
                save_checkpoint(processed_files, all_metadata)
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            # Save checkpoint even on error
            save_checkpoint(processed_files, all_metadata)
            continue
    
    # Final checkpoint save
    save_checkpoint(processed_files, all_metadata)
    
    # Create DataFrame
    df = pd.DataFrame(all_metadata)
    
    # Reorder columns for clarity
    column_order = [
        'datapoint',
        'original_file',
        'chunk_index',
        'duration',
        'sample_rate',
        'raga',
        'thaat',
        'dataset_source',
        'percussion',
        'vocal_instrumental',
        'vocals_path',
        'other_path',
        'instrumental_path',
        'spectral_energy',
        'processing_timestamp',
    ]
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
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
