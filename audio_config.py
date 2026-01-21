"""
Configuration file for audio processing pipeline.

Modify these values to customize the processing behavior.
"""

# Audio Processing Configuration
PROCESSING_CONFIG = {
    # Loudness normalization target in LUFS
    'target_loudness': -18.0,
    
    # Duration of each audio chunk in seconds
    'chunk_duration': 30.0,
    
    # Target sample rate for all audio (Hz)
    # Common values: 16000, 22050, 44100, 48000
    'sample_rate': 22050,
    
    # Energy threshold percentile for filtering
    # Higher value = more aggressive filtering
    # 10 = remove bottom 10% of chunks by energy
    'energy_threshold_percentile': 10,
    
    # Silence trimming threshold in dB below peak
    # Higher value = more aggressive trimming
    'silence_threshold_db': 40,
    
    # Minimum chunk duration to keep (seconds)
    # Chunks shorter than this after trimming are discarded
    'min_chunk_duration': 5.0,
    
    # Supported audio file extensions
    'audio_extensions': ['.mp3', '.wav', '.flac', '.ogg', '.m4a'],
    
    # Output directory for processed files
    'output_dir': 'processed',
    
    # Input data directory
    'data_dir': 'data',
}

# Demucs Configuration
DEMUCS_CONFIG = {
    # Model to use for source separation
    # Options: 'htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'mdx_extra'
    # 'htdemucs' is recommended for best quality
    'model': 'htdemucs',
    
    # Device for processing ('cpu' or 'cuda')
    # Will auto-detect if None
    'device': None,
    
    # Number of random shifts for equivariant stabilization
    # Higher = better quality but slower (0-10)
    'shifts': 1,
    
    # Split audio into smaller chunks for processing
    # Useful for long audio or limited memory
    'split': True,
    
    # Overlap between chunks (0.0-1.0)
    'overlap': 0.25,
    
    # Vocal detection threshold
    # Ratio of vocal energy to other stems energy
    # Lower = more sensitive vocal detection
    'vocal_threshold_ratio': 0.1,
}

# CSV Export Configuration
CSV_CONFIG = {
    # Column order in the output CSV
    'column_order': [
        'datapoint',
        'original_file',
        'chunk_index',
        'duration',
        'sample_rate',
        'percussion',
        'vocal_instrumental',
        'vocals_path',
        'other_path',
        'instrumental_path',
        'spectral_energy',
        'processing_timestamp',
    ],
    
    # Include full absolute paths in CSV
    # If False, uses relative paths
    'use_absolute_paths': False,
}

# Logging Configuration
LOGGING_CONFIG = {
    # Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    'level': 'INFO',
    
    # Log file path (None = no file logging)
    'log_file': None,  # e.g., 'processed/processing.log'
    
    # Show progress bars
    'show_progress': True,
}

# Advanced Configuration
ADVANCED_CONFIG = {
    # Number of parallel workers for processing
    # None = use all available cores
    'num_workers': None,
    
    # Skip files that have already been processed
    'skip_existing': True,
    
    # Save intermediate results after each file
    'save_intermediate': False,
    
    # Validate audio integrity before processing
    'validate_audio': True,
    
    # Maximum audio file size in MB (None = no limit)
    'max_file_size_mb': None,
    
    # Resample method for librosa
    # Options: 'kaiser_best', 'kaiser_fast', 'scipy', 'polyphase'
    'resample_method': 'kaiser_fast',
}


def get_config():
    """
    Get the complete configuration dictionary.
    
    Returns:
        dict: Complete configuration
    """
    return {
        'processing': PROCESSING_CONFIG,
        'demucs': DEMUCS_CONFIG,
        'csv': CSV_CONFIG,
        'logging': LOGGING_CONFIG,
        'advanced': ADVANCED_CONFIG,
    }


def print_config():
    """Print the current configuration."""
    import json
    config = get_config()
    print("=" * 80)
    print("Audio Processing Pipeline Configuration")
    print("=" * 80)
    print(json.dumps(config, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    print_config()
