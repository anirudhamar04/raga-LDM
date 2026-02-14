#!/usr/bin/env python
# coding: utf8

"""
Training helper utilities for Spleeter fine-tuning.

This module provides functions for dataset validation, audio file checking,
and training progress monitoring.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def validate_csv_format(
    csv_path: str,
    required_stems: List[str],
    mix_name: str = "mix"
) -> Tuple[bool, List[str]]:
    """
    Validate the format of a dataset CSV file.
    
    Args:
        csv_path: Path to the CSV file
        required_stems: List of required stem names
        mix_name: Name of the mix column (default: "mix")
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check if file exists
    if not os.path.exists(csv_path):
        errors.append(f"CSV file not found: {csv_path}")
        return False, errors
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        errors.append(f"Failed to read CSV: {e}")
        return False, errors
    
    # Check required columns
    required_columns = [f"{mix_name}_path"] + [f"{stem}_path" for stem in required_stems] + ["duration"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Check for empty dataframe
    if len(df) == 0:
        errors.append("CSV file is empty")
    
    # Check for null values
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        errors.append(f"Found null values in columns: {null_counts[null_counts > 0].to_dict()}")
    
    # Check duration column
    if "duration" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["duration"]):
            errors.append("Duration column must contain numeric values")
        elif (df["duration"] <= 0).any():
            errors.append("Duration values must be positive")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def check_audio_file(
    file_path: str,
    expected_sr: int = 44100,
    min_duration: float = 5.0
) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    Check if an audio file is valid and get its properties.
    
    Args:
        file_path: Path to the audio file
        expected_sr: Expected sample rate (default: 44100)
        min_duration: Minimum required duration in seconds (default: 5.0)
        
    Returns:
        Tuple of (is_valid, error_message, properties)
        properties: Dict with keys 'duration', 'sample_rate', 'channels'
    """
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}", None
    
    try:
        # Get audio info without loading the entire file
        info = sf.info(file_path)
        
        properties = {
            'duration': info.duration,
            'sample_rate': info.samplerate,
            'channels': info.channels
        }
        
        # Validate properties
        if info.duration < min_duration:
            return False, f"Duration too short: {info.duration:.2f}s (minimum: {min_duration}s)", properties
        
        # Note: Sample rate mismatch is not an error, just a warning (will be resampled)
        if info.samplerate != expected_sr:
            logger.debug(f"Sample rate mismatch in {file_path}: {info.samplerate} (expected: {expected_sr})")
        
        return True, None, properties
        
    except Exception as e:
        return False, f"Failed to read audio file: {e}", None


def validate_dataset(
    csv_path: str,
    data_dir: str,
    stems: List[str],
    mix_name: str = "mix",
    sample_rate: int = 44100,
    check_audio_files: bool = True,
    max_files_to_check: Optional[int] = None
) -> Dict:
    """
    Validate an entire dataset including CSV format and audio files.
    
    Args:
        csv_path: Path to the CSV file
        data_dir: Base directory containing audio files
        stems: List of stem names
        mix_name: Name of the mix column (default: "mix")
        sample_rate: Expected sample rate (default: 44100)
        check_audio_files: Whether to check actual audio files (default: True)
        max_files_to_check: Maximum number of audio files to check (None = all)
        
    Returns:
        Dictionary with validation results and statistics
    """
    results = {
        'csv_valid': False,
        'csv_errors': [],
        'num_samples': 0,
        'audio_errors': [],
        'num_audio_checked': 0,
        'num_audio_valid': 0,
        'total_duration': 0.0,
        'duration_stats': {},
        'sample_rate_distribution': {},
        'warnings': []
    }
    
    # Validate CSV format
    csv_valid, csv_errors = validate_csv_format(csv_path, stems, mix_name)
    results['csv_valid'] = csv_valid
    results['csv_errors'] = csv_errors
    
    if not csv_valid:
        logger.error(f"CSV validation failed: {csv_errors}")
        return results
    
    # Load CSV
    df = pd.read_csv(csv_path)
    results['num_samples'] = len(df)
    results['total_duration'] = df['duration'].sum()
    results['duration_stats'] = {
        'mean': df['duration'].mean(),
        'std': df['duration'].std(),
        'min': df['duration'].min(),
        'max': df['duration'].max(),
        'median': df['duration'].median()
    }
    
    logger.info(f"CSV contains {len(df)} samples with total duration {results['total_duration']:.2f}s")
    logger.info(f"Duration stats: mean={results['duration_stats']['mean']:.2f}s, "
                f"median={results['duration_stats']['median']:.2f}s, "
                f"min={results['duration_stats']['min']:.2f}s, "
                f"max={results['duration_stats']['max']:.2f}s")
    
    # Check audio files
    if check_audio_files:
        logger.info("Checking audio files...")
        
        # Determine how many files to check
        num_to_check = len(df) if max_files_to_check is None else min(max_files_to_check, len(df))
        
        # Sample files to check
        if num_to_check < len(df):
            indices_to_check = np.random.choice(len(df), num_to_check, replace=False)
            logger.info(f"Checking random sample of {num_to_check} audio files (out of {len(df)})")
        else:
            indices_to_check = range(len(df))
            logger.info(f"Checking all {num_to_check} audio files")
        
        sample_rates = []
        
        for idx in indices_to_check:
            row = df.iloc[idx]
            
            # Check all stems for this sample
            all_stems = [mix_name] + stems
            for stem in all_stems:
                col_name = f"{stem}_path"
                file_path = os.path.join(data_dir, row[col_name])
                
                is_valid, error_msg, properties = check_audio_file(file_path, sample_rate)
                results['num_audio_checked'] += 1
                
                if is_valid:
                    results['num_audio_valid'] += 1
                    if properties:
                        sample_rates.append(properties['sample_rate'])
                else:
                    results['audio_errors'].append({
                        'sample_index': idx,
                        'stem': stem,
                        'file_path': file_path,
                        'error': error_msg
                    })
        
        # Compute sample rate distribution
        if sample_rates:
            unique_srs, counts = np.unique(sample_rates, return_counts=True)
            results['sample_rate_distribution'] = {int(sr): int(count) for sr, count in zip(unique_srs, counts)}
        
        # Log results
        logger.info(f"Checked {results['num_audio_checked']} audio files: "
                    f"{results['num_audio_valid']} valid, "
                    f"{len(results['audio_errors'])} errors")
        
        if results['sample_rate_distribution']:
            logger.info(f"Sample rate distribution: {results['sample_rate_distribution']}")
        
        # Add warnings for sample rate mismatches
        for sr, count in results['sample_rate_distribution'].items():
            if sr != sample_rate:
                results['warnings'].append(
                    f"{count} files have sample rate {sr}Hz instead of expected {sample_rate}Hz "
                    f"(will be resampled during training)"
                )
    
    return results


def compute_dataset_statistics(
    csv_path: str,
    data_dir: str,
    stems: List[str],
    output_json: Optional[str] = None
) -> Dict:
    """
    Compute detailed statistics about a dataset.
    
    Args:
        csv_path: Path to the CSV file
        data_dir: Base directory containing audio files
        stems: List of stem names
        output_json: Optional path to save statistics as JSON
        
    Returns:
        Dictionary with dataset statistics
    """
    df = pd.read_csv(csv_path)
    
    stats = {
        'num_samples': len(df),
        'total_duration_seconds': float(df['duration'].sum()),
        'total_duration_hours': float(df['duration'].sum() / 3600),
        'duration_statistics': {
            'mean': float(df['duration'].mean()),
            'std': float(df['duration'].std()),
            'min': float(df['duration'].min()),
            'max': float(df['duration'].max()),
            'median': float(df['duration'].median()),
            'q25': float(df['duration'].quantile(0.25)),
            'q75': float(df['duration'].quantile(0.75))
        },
        'stems': stems,
        'csv_path': csv_path,
        'data_dir': data_dir
    }
    
    # Save to JSON if requested
    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to: {output_json}")
    
    return stats


def print_validation_report(results: Dict) -> None:
    """
    Print a formatted validation report.
    
    Args:
        results: Validation results dictionary from validate_dataset()
    """
    print("\n" + "=" * 80)
    print("DATASET VALIDATION REPORT")
    print("=" * 80)
    
    # CSV validation
    print(f"\nCSV Validation: {'✓ PASS' if results['csv_valid'] else '✗ FAIL'}")
    if results['csv_errors']:
        print("  Errors:")
        for error in results['csv_errors']:
            print(f"    - {error}")
    
    # Dataset statistics
    print(f"\nDataset Statistics:")
    print(f"  Number of samples: {results['num_samples']}")
    print(f"  Total duration: {results['total_duration']:.2f}s ({results['total_duration']/3600:.2f}h)")
    
    if results['duration_stats']:
        stats = results['duration_stats']
        print(f"  Duration statistics:")
        print(f"    Mean: {stats['mean']:.2f}s")
        print(f"    Median: {stats['median']:.2f}s")
        print(f"    Std: {stats['std']:.2f}s")
        print(f"    Range: [{stats['min']:.2f}s, {stats['max']:.2f}s]")
    
    # Audio file validation
    if results['num_audio_checked'] > 0:
        print(f"\nAudio File Validation:")
        print(f"  Files checked: {results['num_audio_checked']}")
        print(f"  Valid files: {results['num_audio_valid']}")
        print(f"  Invalid files: {len(results['audio_errors'])}")
        
        if results['sample_rate_distribution']:
            print(f"  Sample rate distribution: {results['sample_rate_distribution']}")
        
        if results['audio_errors']:
            print(f"\n  Errors (showing first 10):")
            for error in results['audio_errors'][:10]:
                print(f"    - Sample {error['sample_index']}, stem '{error['stem']}': {error['error']}")
    
    # Warnings
    if results['warnings']:
        print(f"\nWarnings:")
        for warning in results['warnings']:
            print(f"  ⚠ {warning}")
    
    # Overall status
    print("\n" + "=" * 80)
    overall_valid = results['csv_valid'] and len(results['audio_errors']) == 0
    status = "✓ DATASET IS VALID" if overall_valid else "⚠ DATASET HAS ISSUES"
    print(f"{status}")
    print("=" * 80 + "\n")


def main():
    """Command-line interface for dataset validation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate Spleeter training dataset"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Base directory containing audio files"
    )
    parser.add_argument(
        "--stems",
        type=str,
        nargs='+',
        default=["vocals", "violin", "ghatam", "mridangam", "drone"],
        help="List of stem names"
    )
    parser.add_argument(
        "--mix_name",
        type=str,
        default="mix",
        help="Name of the mix column (default: mix)"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="Expected sample rate (default: 44100)"
    )
    parser.add_argument(
        "--check_audio",
        action="store_true",
        help="Check actual audio files (can be slow)"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of audio files to check (default: all)"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save validation results as JSON"
    )
    
    args = parser.parse_args()
    
    # Run validation
    results = validate_dataset(
        args.csv,
        args.data_dir,
        args.stems,
        args.mix_name,
        args.sample_rate,
        args.check_audio,
        args.max_files
    )
    
    # Print report
    print_validation_report(results)
    
    # Save results
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Validation results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
