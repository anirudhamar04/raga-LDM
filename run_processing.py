#!/usr/bin/env python
"""
Quick start script for audio processing pipeline.

This script provides a simple interface to run the audio processing pipeline
with various options.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Audio Processing Pipeline for Raga-LDM Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all audio files with data and output directories
  python run_processing.py /path/to/data /path/to/output
  
  # Process with Google Drive paths
  python run_processing.py /content/drive/MyDrive/data /content/drive/MyDrive/output
  
  # Resume from checkpoint after interruption
  python run_processing.py /path/to/data /path/to/output --resume
  
  # Change chunk duration to 20 seconds
  python run_processing.py /path/to/data /path/to/output --chunk-duration 20
  
  # More aggressive energy filtering (remove bottom 20%)
  python run_processing.py /path/to/data /path/to/output --energy-threshold 20
  
  # Dry run (show what would be processed)
  python run_processing.py /path/to/data /path/to/output --dry-run
        """
    )
    
    parser.add_argument(
        'data_dir',
        type=str,
        help='Input data directory (required)'
    )
    
    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory (required)'
    )
    
    parser.add_argument(
        '--chunk-duration',
        type=float,
        default=30.0,
        help='Chunk duration in seconds (default: 30.0)'
    )
    
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=22050,
        help='Target sample rate in Hz (default: 22050)'
    )
    
    parser.add_argument(
        '--energy-threshold',
        type=float,
        default=10.0,
        help='Energy threshold percentile for filtering (default: 10.0)'
    )
    
    parser.add_argument(
        '--target-loudness',
        type=float,
        default=-18.0,
        help='Target loudness in LUFS (default: -18.0)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually processing'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous checkpoint if available'
    )
    
    args = parser.parse_args()
    
    # Import here to avoid loading heavy libraries if just showing help
    from process_audio_pipeline import (
        CONFIG, find_audio_files, process_all_files,
        ensure_directories, logger, load_checkpoint
    )
    import logging
    
    # Update configuration
    CONFIG['data_dir'] = str(Path(args.data_dir).resolve())
    CONFIG['output_dir'] = str(Path(args.output_dir).resolve())
    CONFIG['chunk_duration'] = args.chunk_duration
    CONFIG['sample_rate'] = args.sample_rate
    CONFIG['energy_threshold_percentile'] = args.energy_threshold
    CONFIG['target_loudness'] = args.target_loudness
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Print configuration
    print("=" * 80)
    print("Audio Processing Pipeline Configuration")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Chunk duration: {args.chunk_duration} seconds")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Energy threshold: {args.energy_threshold}%")
    print(f"Target loudness: {args.target_loudness} LUFS")
    print("=" * 80)
    
    # Check if data directory exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"\nError: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Find audio files
    audio_files = find_audio_files(str(data_path))
    
    if not audio_files:
        print(f"\nNo audio files found in {args.data_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(audio_files)} audio files")
    
    if args.dry_run:
        print("\nDry run - files that would be processed:")
        for i, file in enumerate(audio_files[:10], 1):
            print(f"  {i}. {file}")
        if len(audio_files) > 10:
            print(f"  ... and {len(audio_files) - 10} more files")
        print("\nRun without --dry-run to actually process these files.")
        sys.exit(0)
    
    # Ensure directories exist
    ensure_directories()
    
    # Check for existing checkpoint
    if args.resume:
        checkpoint = load_checkpoint()
        if checkpoint['processed_files']:
            print(f"\n✓ Found checkpoint with {len(checkpoint['processed_files'])} processed files")
            print(f"  Last updated: {checkpoint.get('last_updated', 'unknown')}")
        else:
            print("\n⚠ No checkpoint found, starting fresh")
    
    # Run the pipeline
    metadata_df = process_all_files(str(data_path), resume=args.resume)
    
    if metadata_df.empty:
        print("\nNo data was processed!")
        sys.exit(1)
    
    # Save metadata
    output_csv = Path(args.output_dir) / 'metadata.csv'
    metadata_df.to_csv(output_csv, index=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print(f"Total chunks: {len(metadata_df)}")
    print(f"Vocal clips: {(metadata_df['vocal_instrumental'] == 'vocal').sum()}")
    print(f"Instrumental clips: {(metadata_df['vocal_instrumental'] == 'instrumental').sum()}")
    print(f"Total duration: {metadata_df['duration'].sum() / 3600:.2f} hours")
    print(f"Metadata saved to: {output_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
