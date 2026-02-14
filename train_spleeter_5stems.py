#!/usr/bin/env python
# coding: utf8

"""
Training script for fine-tuning Spleeter on 5-stem Carnatic music separation.

This script trains a Spleeter model to separate audio into 5 stems:
- vocals
- violin
- ghatam
- mridangam
- drone

Usage:
    python train_spleeter_5stems.py \
        --config spleeter/configs/5stems_carnatic/base_config.json \
        --data /path/to/dataset \
        --pretrained_model spleeter:4stems
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from functools import partial
from typing import Optional

import tensorflow as tf

# Add spleeter to path
sys.path.insert(0, str(Path(__file__).parent / "spleeter"))

from spleeter.audio.adapter import AudioAdapter
from spleeter.dataset import get_training_dataset, get_validation_dataset
from spleeter.model import model_fn
from spleeter.model.provider import ModelProvider
from spleeter.utils.configuration import load_configuration

# Import custom utilities
sys.path.insert(0, str(Path(__file__).parent))
from utils.training_helpers import validate_dataset, print_validation_report, compute_dataset_statistics
from utils.init_pretrained_weights import initialize_from_pretrained

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Spleeter model for 5-stem separation"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Base directory containing audio files"
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="Pretrained model to fine-tune from (e.g., 'spleeter:4stems')"
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate the dataset without training"
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip dataset validation before training"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing checkpoint"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="spleeter.audio.ffmpeg.FFMPEGProcessAudioAdapter",
        help="Audio adapter to use"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--gpu_memory_fraction",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use (default: 0.9)"
    )
    
    return parser.parse_args()


def validate_configuration(params: dict) -> bool:
    """
    Validate the configuration parameters.
    
    Args:
        params: Configuration dictionary
        
    Returns:
        True if configuration is valid, False otherwise
    """
    required_keys = [
        "train_csv",
        "validation_csv",
        "model_dir",
        "instrument_list",
        "sample_rate",
        "frame_length",
        "frame_step",
        "T",
        "F",
        "n_channels"
    ]
    
    missing_keys = [key for key in required_keys if key not in params]
    if missing_keys:
        logger.error(f"Missing required configuration keys: {missing_keys}")
        return False
    
    # Check instrument list
    if len(params["instrument_list"]) != 5:
        logger.warning(
            f"Expected 5 instruments, got {len(params['instrument_list'])}: "
            f"{params['instrument_list']}"
        )
    
    return True


def setup_model_directory(
    params: dict,
    pretrained_model: Optional[str],
    resume: bool
) -> str:
    """
    Set up the model directory for training.
    
    Args:
        params: Configuration dictionary
        pretrained_model: Name of pretrained model to load (if any)
        resume: Whether to resume from existing checkpoint
        
    Returns:
        Path to model directory
    """
    model_dir = params["model_dir"]
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(model_dir, "params.json")
    with open(config_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved configuration to: {config_path}")
    
    # Initialize from pretrained model if requested
    if pretrained_model and not resume:
        logger.info(f"Initializing from pretrained model: {pretrained_model}")
        
        # Check if checkpoint already exists
        checkpoint_path = tf.train.latest_checkpoint(model_dir)
        if checkpoint_path:
            logger.warning(
                f"Model directory already contains checkpoint: {checkpoint_path}"
            )
            response = input("Overwrite existing checkpoint? (y/n): ").strip().lower()
            if response != 'y':
                logger.info("Using existing checkpoint")
                return model_dir
        
        # Initialize from pretrained
        try:
            from utils.init_pretrained_weights import initialize_from_pretrained
            
            temp_config = config_path
            initialized_dir = initialize_from_pretrained(
                pretrained_model,
                temp_config,
                model_dir
            )
            logger.info(f"Initialized model from {pretrained_model}")
        except Exception as e:
            logger.warning(f"Failed to initialize from pretrained model: {e}")
            logger.warning("Training will start with random initialization")
    
    elif resume:
        checkpoint_path = tf.train.latest_checkpoint(model_dir)
        if checkpoint_path:
            logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        else:
            logger.warning(f"No checkpoint found in {model_dir}, starting from scratch")
    
    return model_dir


def train(
    params: dict,
    data_dir: str,
    adapter: str,
    gpu_memory_fraction: float = 0.9
) -> None:
    """
    Train the Spleeter model.
    
    Args:
        params: Configuration dictionary
        data_dir: Base directory containing audio files
        adapter: Audio adapter name
        gpu_memory_fraction: Fraction of GPU memory to use
    """
    # Get audio adapter
    audio_adapter = AudioAdapter.get(adapter)
    
    # Configure TensorFlow session
    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    session_config.gpu_options.allow_growth = True
    
    # Create estimator
    logger.info("Creating TensorFlow estimator...")
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=params["model_dir"],
        params=params,
        config=tf.estimator.RunConfig(
            save_checkpoints_steps=params.get("save_checkpoints_steps", 500),
            tf_random_seed=params.get("random_seed", 42),
            save_summary_steps=params.get("save_summary_steps", 10),
            session_config=session_config,
            log_step_count_steps=50,
            keep_checkpoint_max=5,
        ),
    )
    
    # Create training input function
    logger.info("Creating training dataset...")
    train_input_fn = partial(
        get_training_dataset,
        params,
        audio_adapter,
        data_dir
    )
    
    # Create validation input function
    logger.info("Creating validation dataset...")
    eval_input_fn = partial(
        get_validation_dataset,
        params,
        audio_adapter,
        data_dir
    )
    
    # Create train and eval specs
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=params.get("train_max_steps", 2000000)
    )
    
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=None,
        throttle_secs=params.get("throttle_secs", 600)
    )
    
    # Train the model
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info(f"Model directory: {params['model_dir']}")
    logger.info(f"Instruments: {params['instrument_list']}")
    logger.info(f"Max steps: {params.get('train_max_steps', 2000000)}")
    logger.info(f"Batch size: {params.get('batch_size', 4)}")
    logger.info(f"Learning rate: {params.get('learning_rate', 5e-5)}")
    logger.info("=" * 80)
    
    try:
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        logger.info("Training completed successfully!")
        
        # Write model probe
        ModelProvider.writeProbe(params["model_dir"])
        logger.info(f"Model probe written to {params['model_dir']}")
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info(f"Checkpoint saved in: {params['model_dir']}")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


def main():
    """Main execution function."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    params = load_configuration(args.config)
    
    # Validate configuration
    if not validate_configuration(params):
        logger.error("Configuration validation failed")
        sys.exit(1)
    
    # Update CSV paths to be absolute
    train_csv = params["train_csv"]
    validation_csv = params["validation_csv"]
    
    if not os.path.isabs(train_csv):
        train_csv = os.path.join(args.data, train_csv)
        params["train_csv"] = train_csv
    
    if not os.path.isabs(validation_csv):
        validation_csv = os.path.join(args.data, validation_csv)
        params["validation_csv"] = validation_csv
    
    logger.info(f"Training CSV: {train_csv}")
    logger.info(f"Validation CSV: {validation_csv}")
    logger.info(f"Data directory: {args.data}")
    
    # Validate dataset
    if not args.skip_validation:
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATING TRAINING DATASET")
        logger.info("=" * 80)
        
        train_results = validate_dataset(
            train_csv,
            args.data,
            params["instrument_list"],
            params.get("mix_name", "mix"),
            params["sample_rate"],
            check_audio_files=True,
            max_files_to_check=100  # Check first 100 files
        )
        print_validation_report(train_results)
        
        if not train_results["csv_valid"]:
            logger.error("Training dataset validation failed!")
            sys.exit(1)
        
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATING VALIDATION DATASET")
        logger.info("=" * 80)
        
        val_results = validate_dataset(
            validation_csv,
            args.data,
            params["instrument_list"],
            params.get("mix_name", "mix"),
            params["sample_rate"],
            check_audio_files=True,
            max_files_to_check=50  # Check first 50 files
        )
        print_validation_report(val_results)
        
        if not val_results["csv_valid"]:
            logger.error("Validation dataset validation failed!")
            sys.exit(1)
        
        # Compute and save statistics
        logger.info("Computing dataset statistics...")
        train_stats = compute_dataset_statistics(
            train_csv,
            args.data,
            params["instrument_list"],
            output_json=os.path.join(params["model_dir"], "train_statistics.json")
        )
        val_stats = compute_dataset_statistics(
            validation_csv,
            args.data,
            params["instrument_list"],
            output_json=os.path.join(params["model_dir"], "validation_statistics.json")
        )
        
        logger.info(f"Training set: {train_stats['num_samples']} samples, "
                    f"{train_stats['total_duration_hours']:.2f} hours")
        logger.info(f"Validation set: {val_stats['num_samples']} samples, "
                    f"{val_stats['total_duration_hours']:.2f} hours")
    
    # Exit if validation only
    if args.validate_only:
        logger.info("\nValidation complete. Exiting (--validate_only flag set)")
        return
    
    # Set up model directory
    logger.info("\nSetting up model directory...")
    model_dir = setup_model_directory(
        params,
        args.pretrained_model,
        args.resume
    )
    params["model_dir"] = model_dir
    
    # Train the model
    train(
        params,
        args.data,
        args.adapter,
        args.gpu_memory_fraction
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Model saved to: {model_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
