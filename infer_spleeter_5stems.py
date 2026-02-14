#!/usr/bin/env python
# coding: utf8

"""
Inference script for 5-stem Carnatic music separation using trained Spleeter model.

This script uses a trained Spleeter model to separate audio files into 5 stems:
- vocals
- violin
- ghatam
- mridangam
- drone

Usage:
    # Separate a single file
    python infer_spleeter_5stems.py \
        --model_dir trained_models/5stems_carnatic \
        --input audio_file.wav \
        --output output_dir/

    # Separate multiple files
    python infer_spleeter_5stems.py \
        --model_dir trained_models/5stems_carnatic \
        --input audio1.wav audio2.wav audio3.wav \
        --output output_dir/
    
    # Separate all files in a directory
    python infer_spleeter_5stems.py \
        --model_dir trained_models/5stems_carnatic \
        --input_dir audio_files/ \
        --output output_dir/
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional
from glob import glob

# Add spleeter to path
sys.path.insert(0, str(Path(__file__).parent / "spleeter"))

from spleeter.separator import Separator
from spleeter.audio import Codec
from spleeter.audio.adapter import AudioAdapter

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Separate audio files using trained 5-stem Spleeter model"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing trained model checkpoint"
    )
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=str,
        nargs='+',
        help="Input audio file(s) to separate"
    )
    input_group.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing audio files to separate"
    )
    
    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for separated stems"
    )
    parser.add_argument(
        "--filename_format",
        type=str,
        default="{filename}/{instrument}.{codec}",
        help="Output filename format (default: {filename}/{instrument}.{codec})"
    )
    
    # Audio processing arguments
    parser.add_argument(
        "--codec",
        type=str,
        default="wav",
        choices=["wav", "mp3", "ogg", "m4a", "flac"],
        help="Output audio codec (default: wav)"
    )
    parser.add_argument(
        "--bitrate",
        type=str,
        default="128k",
        help="Output bitrate for lossy codecs (default: 128k)"
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Start offset in seconds (default: 0.0)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration to process in seconds (default: full file)"
    )
    
    # Processing options
    parser.add_argument(
        "--adapter",
        type=str,
        default="spleeter.audio.ffmpeg.FFMPEGProcessAudioAdapter",
        help="Audio adapter to use"
    )
    parser.add_argument(
        "--mwf",
        action="store_true",
        help="Use multichannel Wiener filtering for better separation quality (slower)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process input directory recursively"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs='+',
        default=['.wav', '.mp3', '.flac', '.ogg', '.m4a'],
        help="File extensions to process when using --input_dir"
    )
    
    return parser.parse_args()


def find_audio_files(
    directory: str,
    extensions: List[str],
    recursive: bool = False
) -> List[str]:
    """
    Find all audio files in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include
        recursive: Whether to search recursively
        
    Returns:
        List of audio file paths
    """
    audio_files = []
    
    if recursive:
        for ext in extensions:
            pattern = os.path.join(directory, "**", f"*{ext}")
            audio_files.extend(glob(pattern, recursive=True))
    else:
        for ext in extensions:
            pattern = os.path.join(directory, f"*{ext}")
            audio_files.extend(glob(pattern))
    
    return sorted(audio_files)


def verify_model_directory(model_dir: str) -> bool:
    """
    Verify that the model directory contains necessary files.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        True if valid, False otherwise
    """
    # Check if directory exists
    if not os.path.exists(model_dir):
        logger.error(f"Model directory not found: {model_dir}")
        return False
    
    # Check for params.json
    params_path = os.path.join(model_dir, "params.json")
    if not os.path.exists(params_path):
        logger.error(f"Configuration file not found: {params_path}")
        return False
    
    # Check for checkpoint
    checkpoint = tf.train.latest_checkpoint(model_dir)
    if checkpoint is None:
        logger.error(f"No checkpoint found in: {model_dir}")
        return False
    
    logger.info(f"Found checkpoint: {checkpoint}")
    
    return True


def load_model_config(model_dir: str) -> dict:
    """
    Load model configuration.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        Configuration dictionary
    """
    params_path = os.path.join(model_dir, "params.json")
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    logger.info(f"Loaded configuration from: {params_path}")
    logger.info(f"Model instruments: {params.get('instrument_list', [])}")
    
    return params


def separate_audio_files(
    input_files: List[str],
    model_dir: str,
    output_dir: str,
    adapter: str,
    codec: str,
    bitrate: str,
    offset: float,
    duration: Optional[float],
    filename_format: str,
    mwf: bool
) -> None:
    """
    Separate multiple audio files.
    
    Args:
        input_files: List of input audio file paths
        model_dir: Path to trained model directory
        output_dir: Output directory for separated stems
        adapter: Audio adapter name
        codec: Output audio codec
        bitrate: Output bitrate
        offset: Start offset in seconds
        duration: Duration to process (None = full file)
        filename_format: Output filename format
        mwf: Whether to use multichannel Wiener filtering
    """
    # Get audio adapter
    audio_adapter = AudioAdapter.get(adapter)
    
    # Create separator
    logger.info(f"Loading model from: {model_dir}")
    separator = Separator(model_dir, MWF=mwf)
    
    # Process each file
    logger.info(f"Processing {len(input_files)} audio file(s)...")
    
    for i, input_file in enumerate(input_files, 1):
        logger.info(f"\n[{i}/{len(input_files)}] Processing: {input_file}")
        
        try:
            # Perform separation
            separator.separate_to_file(
                input_file,
                output_dir,
                audio_adapter=audio_adapter,
                offset=offset,
                duration=duration,
                codec=Codec[codec.upper()],
                bitrate=bitrate,
                filename_format=filename_format,
                synchronous=True  # Process synchronously for better error handling
            )
            
            logger.info(f"✓ Successfully separated: {input_file}")
            
        except Exception as e:
            logger.error(f"✗ Failed to separate {input_file}: {e}")
            continue
    
    logger.info(f"\nSeparation complete! Output saved to: {output_dir}")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Verify model directory
    logger.info("Verifying model directory...")
    if not verify_model_directory(args.model_dir):
        logger.error("Model verification failed")
        sys.exit(1)
    
    # Load model configuration
    model_config = load_model_config(args.model_dir)
    
    # Get input files
    if args.input:
        input_files = args.input
    else:
        logger.info(f"Searching for audio files in: {args.input_dir}")
        input_files = find_audio_files(
            args.input_dir,
            args.extensions,
            args.recursive
        )
        
        if not input_files:
            logger.error(f"No audio files found in: {args.input_dir}")
            sys.exit(1)
        
        logger.info(f"Found {len(input_files)} audio file(s)")
    
    # Verify input files exist
    missing_files = [f for f in input_files if not os.path.exists(f)]
    if missing_files:
        logger.error(f"Input files not found: {missing_files}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    logger.info(f"Output directory: {args.output}")
    
    # Log separation settings
    logger.info("\n" + "=" * 80)
    logger.info("SEPARATION SETTINGS")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_dir}")
    logger.info(f"Stems: {model_config.get('instrument_list', [])}")
    logger.info(f"Input files: {len(input_files)}")
    logger.info(f"Output codec: {args.codec}")
    logger.info(f"Bitrate: {args.bitrate}")
    logger.info(f"MWF: {args.mwf}")
    if args.offset > 0:
        logger.info(f"Offset: {args.offset}s")
    if args.duration:
        logger.info(f"Duration: {args.duration}s")
    logger.info("=" * 80 + "\n")
    
    # Perform separation
    try:
        import tensorflow as tf
        
        separate_audio_files(
            input_files,
            args.model_dir,
            args.output,
            args.adapter,
            args.codec,
            args.bitrate,
            args.offset,
            args.duration,
            args.filename_format,
            args.mwf
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ SEPARATION COMPLETE")
        logger.info(f"Separated {len(input_files)} file(s)")
        logger.info(f"Output location: {args.output}")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("\nSeparation interrupted by user")
    except Exception as e:
        logger.error(f"\nSeparation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
