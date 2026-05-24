#!/usr/bin/env python
# coding: utf8

"""
Utility for initializing pretrained Spleeter weights for fine-tuning.

This module provides functionality to load pretrained Spleeter models
and map their weights to a new model with different stem names.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import tensorflow as tf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_pretrained_model_path(model_name: str = "spleeter:4stems") -> str:
    """
    Get the path to a pretrained Spleeter model.
    
    Args:
        model_name: Name of the pretrained model (e.g., "spleeter:4stems", "spleeter:5stems")
        
    Returns:
        Path to the pretrained model directory
    """
    from spleeter.model.provider import ModelProvider
    
    # Extract the actual model name (e.g., "4stems" from "spleeter:4stems")
    if ":" in model_name:
        actual_model_name = model_name.split(":", 1)[1]
    else:
        actual_model_name = model_name
    
    logger.info(f"Downloading pretrained model: {actual_model_name}")
    model_path = ModelProvider.default().get(actual_model_name)
    logger.info(f"Pretrained model path: {model_path}")
    
    return model_path


def create_stem_mapping(
    pretrained_stems: List[str],
    target_stems: List[str]
) -> Dict[str, str]:
    """
    Create a mapping from pretrained stems to target stems.
    
    Args:
        pretrained_stems: List of stems in the pretrained model
        target_stems: List of stems in the target model
        
    Returns:
        Dictionary mapping target stems to pretrained stems
    """
    # Default mapping strategy
    mapping = {}
    
    # Direct mappings where stem names match
    if "vocals" in pretrained_stems and "vocals" in target_stems:
        mapping["vocals"] = "vocals"
    
    # Map percussion-related stems
    if "drums" in pretrained_stems:
        if "mridangam" in target_stems:
            mapping["mridangam"] = "drums"
        if "ghatam" in target_stems and "ghatam" not in mapping:
            # Use drums for ghatam too (both are percussion)
            mapping["ghatam"] = "drums"
    
    # Map melodic instruments - prioritize piano for violin
    if "violin" in target_stems:
        if "piano" in pretrained_stems:
            mapping["violin"] = "piano"
        elif "other" in pretrained_stems:
            mapping["violin"] = "other"
        elif "bass" in pretrained_stems:
            mapping["violin"] = "bass"
    
    # Map drone - prioritize bass
    if "drone" in target_stems:
        if "bass" in pretrained_stems:
            mapping["drone"] = "bass"
        elif "other" in pretrained_stems:
            mapping["drone"] = "other"
    
    # Map remaining percussion stems
    if "drums" in pretrained_stems:
        # If we have multiple percussion targets, try to map them intelligently
        percussion_targets = [s for s in target_stems if s in ["mridangam", "ghatam", "drums"]]
        if len(percussion_targets) > 1:
            # Map drums to the first unmapped percussion target
            for target in percussion_targets:
                if target not in mapping:
                    mapping[target] = "drums"
                    break
    
    logger.info(f"Created stem mapping: {mapping}")
    return mapping


def get_checkpoint_variables(checkpoint_path: str) -> Dict[str, Tuple]:
    """
    Get all variables from a TensorFlow checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        
    Returns:
        Dictionary of variable names to (shape, dtype) tuples
    """
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
    if checkpoint_file is None:
        raise ValueError(f"No checkpoint found in {checkpoint_path}")
    
    reader = tf.train.load_checkpoint(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_to_dtype_map = reader.get_variable_to_dtype_map()
    
    variables = {}
    for var_name in var_to_shape_map:
        variables[var_name] = (var_to_shape_map[var_name], var_to_dtype_map[var_name])
    
    return variables


def map_checkpoint_variables(
    checkpoint_path: str,
    stem_mapping: Dict[str, str],
    output_checkpoint_path: str
) -> None:
    """
    Map checkpoint variables from pretrained model to new model.
    
    Args:
        checkpoint_path: Path to the pretrained checkpoint
        stem_mapping: Mapping from target stems to pretrained stems
        output_checkpoint_path: Path to save the mapped checkpoint
    """
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
    if checkpoint_file is None:
        raise ValueError(f"No checkpoint found in {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from: {checkpoint_file}")
    
    # Create output directory
    os.makedirs(output_checkpoint_path, exist_ok=True)
    
    # Read the checkpoint
    reader = tf.train.load_checkpoint(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    
    # Create new checkpoint with mapped variables
    with tf.compat.v1.Session() as sess:
        variables_to_save = []
        
        for var_name in var_to_shape_map:
            # Load variable value
            tensor_value = reader.get_tensor(var_name)
            
            # Check if this variable needs to be mapped
            mapped_name = var_name
            for target_stem, source_stem in stem_mapping.items():
                if f"/{source_stem}_" in var_name:
                    mapped_name = var_name.replace(f"/{source_stem}_", f"/{target_stem}_")
                    logger.debug(f"Mapping variable: {var_name} -> {mapped_name}")
                    break
            
            # Create new variable
            var = tf.Variable(tensor_value, name=mapped_name.split(':')[0])
            variables_to_save.append(var)
        
        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Save checkpoint
        saver = tf.compat.v1.train.Saver(variables_to_save)
        save_path = os.path.join(output_checkpoint_path, "model.ckpt")
        saver.save(sess, save_path)
        
        logger.info(f"Saved mapped checkpoint to: {save_path}")


def initialize_from_pretrained(
    pretrained_model: Optional[str],
    target_config_path: str,
    output_model_dir: str,
    stem_mapping: Optional[Dict[str, str]] = None
) -> str:
    """
    Initialize a new model from a pretrained Spleeter model.
    
    Args:
        pretrained_model: Name or path of the pretrained model. If None, auto-detects based on target stems.
        target_config_path: Path to the target model configuration
        output_model_dir: Directory to save the initialized model
        stem_mapping: Optional custom stem mapping. If None, automatic mapping is used.
        
    Returns:
        Path to the initialized model directory
    """
    # Load target configuration
    with open(target_config_path, 'r') as f:
        target_config = json.load(f)
    
    target_stems = target_config['instrument_list']
    num_target_stems = len(target_stems)
    
    # Auto-detect pretrained model if not specified
    if pretrained_model is None or pretrained_model == "":
        if num_target_stems == 5:
            pretrained_model = "spleeter:5stems"
            logger.info(f"Auto-detected 5 stems in target, using {pretrained_model}")
        elif num_target_stems == 4:
            pretrained_model = "spleeter:4stems"
            logger.info(f"Auto-detected 4 stems in target, using {pretrained_model}")
        elif num_target_stems == 2:
            pretrained_model = "spleeter:2stems"
            logger.info(f"Auto-detected 2 stems in target, using {pretrained_model}")
        else:
            # Default to 5stems for better initialization
            pretrained_model = "spleeter:5stems"
            logger.info(f"Using default {pretrained_model} for {num_target_stems} stems")
    
    # Get pretrained model path
    if pretrained_model.startswith("spleeter:"):
        pretrained_path = get_pretrained_model_path(pretrained_model)
    else:
        pretrained_path = pretrained_model
    
    # Load pretrained configuration
    pretrained_config_path = os.path.join(pretrained_path, "params.json")
    if os.path.exists(pretrained_config_path):
        with open(pretrained_config_path, 'r') as f:
            pretrained_config = json.load(f)
        pretrained_stems = pretrained_config.get('instrument_list', [])
    else:
        # Infer from model name
        if "4stems" in pretrained_model:
            pretrained_stems = ["vocals", "drums", "bass", "other"]
        elif "5stems" in pretrained_model:
            pretrained_stems = ["vocals", "piano", "drums", "bass", "other"]
        else:
            pretrained_stems = ["vocals", "accompaniment"]
    
    logger.info(f"Pretrained stems: {pretrained_stems}")
    logger.info(f"Target stems: {target_stems}")
    
    # Create stem mapping
    if stem_mapping is None:
        stem_mapping = create_stem_mapping(pretrained_stems, target_stems)
    
    # Create output directory
    os.makedirs(output_model_dir, exist_ok=True)
    
    # Copy configuration to output directory
    output_config_path = os.path.join(output_model_dir, "params.json")
    # Only copy if source and destination are different
    if os.path.abspath(target_config_path) != os.path.abspath(output_config_path):
        shutil.copy(target_config_path, output_config_path)
        logger.info(f"Copied configuration to: {output_config_path}")
    else:
        logger.info(f"Configuration already at target location: {output_config_path}")
    
    # Map checkpoint variables
    logger.info("Mapping checkpoint variables...")
    try:
        map_checkpoint_variables(pretrained_path, stem_mapping, output_model_dir)
        logger.info("Successfully mapped checkpoint variables")
    except Exception as e:
        logger.warning(f"Could not map checkpoint variables: {e}")
        logger.warning("Model will be initialized with random weights")
    
    return output_model_dir


def main():
    """Command-line interface for initializing pretrained weights."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Initialize Spleeter model from pretrained weights"
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="Pretrained model name or path (default: auto-detect based on target config, prefers 5stems for 5-stem models)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to target model configuration (JSON)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save initialized model"
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default=None,
        help="Custom stem mapping as JSON string (e.g., '{\"mridangam\": \"drums\"}')"
    )
    
    args = parser.parse_args()
    
    # Parse custom mapping if provided
    stem_mapping = None
    if args.mapping:
        stem_mapping = json.loads(args.mapping)
    
    # Initialize model
    output_path = initialize_from_pretrained(
        args.pretrained_model,
        args.config,
        args.output_dir,
        stem_mapping
    )
    
    print(f"\nInitialized model saved to: {output_path}")
    print("You can now use this model for fine-tuning.")


if __name__ == "__main__":
    main()
