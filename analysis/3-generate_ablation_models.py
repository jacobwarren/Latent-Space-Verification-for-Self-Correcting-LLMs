#!/usr/bin/env python
"""
Generate ablation variants of a verification-enhanced model to test 
which components contribute most to performance improvements.

This script creates the following ablation variants:
1. No cross-layer verification
2. Fixed confidence thresholds (no learned confidence)
3. Single layer verification (at different positions)
4. Shallow vs deep verification (early vs late layers only)
"""

import os
import torch
import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def disable_cross_layer_verification(model, save_path):
    """
    Create a variant with cross-layer verification disabled
    """
    logger.info("Creating model variant with cross-layer verification disabled")

    # Check if model has cross_layer_verifier attribute
    if hasattr(model, 'cross_layer_verifier'):
        # Save original cross_layer_verifier
        original_verifier = model.cross_layer_verifier

        # Replace with dummy module that has no effect
        class DummyVerifier(nn.Module):
            def forward(self, all_hidden_states):
                # Return a tensor of ones as high confidence
                return torch.ones(all_hidden_states[0].shape[0], 1).to(all_hidden_states[0].device)

        model.cross_layer_verifier = DummyVerifier()
        model.enable_cross_layer = False

        # Save the model
        model.save_pretrained(save_path, safe_serialization=False)
        logger.info(f"Model with disabled cross-layer verification saved to {save_path}")

        # Restore original for other ablations
        model.cross_layer_verifier = original_verifier
        model.enable_cross_layer = True
        return True
    else:
        logger.warning("Model does not have cross_layer_verifier attribute")
        return False

def fixed_confidence_thresholds(model, save_path, threshold=0.5):
    """
    Create a variant where confidence scores are fixed rather than learned
    """
    logger.info(f"Creating model variant with fixed confidence threshold: {threshold}")

    # Check if model has verification_adapters
    if not hasattr(model, 'verification_adapters'):
        logger.warning("Model does not have verification_adapters")
        return False

    # Clone model for this ablation
    model_copy = model

    # Replace confidence scorers in all adapters with fixed value
    for adapter_name, adapter in model_copy.verification_adapters.items():
        # Different adapter types have different confidence mechanisms
        if hasattr(adapter, 'confidence_scorer'):
            # Simple case with single confidence scorer
            class FixedConfidence(nn.Module):
                def forward(self, x):
                    return torch.ones_like(x) * threshold

            adapter.confidence_scorer = FixedConfidence()
            logger.info(f"Replaced confidence scorer in {adapter_name}")

        elif hasattr(adapter, 'confidence_mean') and hasattr(adapter, 'confidence_logvar'):
            # Bayesian case with mean and logvar
            class FixedMean(nn.Module):
                def forward(self, x):
                    return torch.ones_like(x) * torch.logit(torch.tensor(threshold))

            class FixedLogVar(nn.Module):
                def forward(self, x):
                    return torch.ones_like(x) * -5.0  # Small variance

            adapter.confidence_mean = FixedMean()
            adapter.confidence_logvar = FixedLogVar()
            logger.info(f"Replaced Bayesian confidence in {adapter_name}")

    # Save the model
    model_copy.save_pretrained(save_path, safe_serialization=False)
    logger.info(f"Model with fixed confidence thresholds saved to {save_path}")
    return True

def single_layer_verification(model, save_path, layer_idx=None):
    """
    Create a variant with verification at only a single layer
    If layer_idx is None, will create variants for each layer separately
    """
    if not hasattr(model, 'verification_adapters') or not hasattr(model, 'adapter_locations'):
        logger.warning("Model does not have verification_adapters or adapter_locations")
        return False

    adapter_locations = model.adapter_locations
    if layer_idx is not None:
        # Only create variant for the specified layer
        if layer_idx not in adapter_locations:
            logger.warning(f"Layer {layer_idx} is not in adapter_locations: {adapter_locations}")
            return False

        layers_to_ablate = [layer_idx]
        output_path = save_path
    else:
        # Create variants for each layer
        layers_to_ablate = adapter_locations
        # Parent output directory
        os.makedirs(save_path, exist_ok=True)

    success = False

    # Create variants for each specified layer
    for layer in layers_to_ablate:
        # Create a model with only this layer having verification
        if layer_idx is None:
            # Set layer-specific output path if generating multiple variants
            output_path = os.path.join(save_path, f"layer_{layer}")
            os.makedirs(output_path, exist_ok=True)

        logger.info(f"Creating model variant with verification only at layer {layer}")

        # Clone model for this ablation
        model_copy = model

        # Save original adapters
        original_adapters = model_copy.verification_adapters

        # Create new adapters dictionary with only the specified layer
        single_layer_adapters = nn.ModuleDict()
        for adapter_idx in adapter_locations:
            if adapter_idx == layer:
                # Keep this layer's adapter
                adapter_name = f"layer_{adapter_idx}"
                if adapter_name in original_adapters:
                    single_layer_adapters[adapter_name] = original_adapters[adapter_name]

        # Replace adapters
        model_copy.verification_adapters = single_layer_adapters
        model_copy.adapter_locations = [layer]

        # Save the model
        model_copy.save_pretrained(output_path, safe_serialization=False)
        logger.info(f"Model with single layer verification at layer {layer} saved to {output_path}")
        success = True

        # Restore original adapters for next iteration
        model_copy.verification_adapters = original_adapters
        model_copy.adapter_locations = adapter_locations

    return success

def early_late_verification(model, save_path):
    """
    Create early-layers and late-layers variants with improved saving
    """
    if not hasattr(model, 'verification_adapters') or not hasattr(model, 'adapter_locations'):
        logger.warning("Model does not have verification_adapters or adapter_locations")
        return False

    adapter_locations = model.adapter_locations
    if len(adapter_locations) <= 2:
        logger.warning(f"Not enough adapter locations for early/late split: {adapter_locations}")
        return False

    # Create paths
    early_count = len(adapter_locations) // 2
    early_layers = adapter_locations[:early_count]
    early_path = os.path.join(save_path, "early_layers")
    os.makedirs(early_path, exist_ok=True)

    late_layers = adapter_locations[early_count:]
    late_path = os.path.join(save_path, "late_layers")
    os.makedirs(late_path, exist_ok=True)

    logger.info(f"Creating early layers variant with adapters at: {early_layers}")
    logger.info(f"Creating late layers variant with adapters at: {late_layers}")

    # Save original configuration
    original_adapters = model.verification_adapters
    original_locations = model.adapter_locations

    try:
        # Memory management before saving
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Process early layers
        early_adapters = nn.ModuleDict()
        for layer in early_layers:
            adapter_name = f"layer_{layer}"
            if adapter_name in original_adapters:
                early_adapters[adapter_name] = original_adapters[adapter_name]

        # Apply early layers configuration
        model.verification_adapters = early_adapters
        model.adapter_locations = early_layers

        # Save early layers model using safetensors
        try:
            model.save_pretrained(early_path, safe_serialization=True)
            logger.info(f"Early layers model saved to {early_path}")
        except Exception as e:
            logger.error(f"Failed to save early layers model: {e}")
            # Fallback to state dict saving
            try:
                logger.info("Trying fallback to state dict saving")
                torch.save(model.state_dict(), os.path.join(early_path, "model_state_dict.pt"))
                # Save verification config separately
                with open(os.path.join(early_path, "verification_config.json"), "w") as f:
                    json.dump({
                        "adapter_locations": early_layers,
                        "hidden_size": model.hidden_size if hasattr(model, "hidden_size") else None,
                        "enable_cross_layer": model.enable_cross_layer if hasattr(model, "enable_cross_layer") else False,
                    }, f)
                logger.info(f"Early layers model state dict saved to {early_path}")
            except Exception as e2:
                logger.error(f"Failed to save state dict: {e2}")

        # Process late layers
        late_adapters = nn.ModuleDict()
        for layer in late_layers:
            adapter_name = f"layer_{layer}"
            if adapter_name in original_adapters:
                late_adapters[adapter_name] = original_adapters[adapter_name]

        # Apply late layers configuration
        model.verification_adapters = late_adapters
        model.adapter_locations = late_layers

        # Save late layers model using safetensors
        try:
            model.save_pretrained(late_path, safe_serialization=True)
            logger.info(f"Late layers model saved to {late_path}")
        except Exception as e:
            logger.error(f"Failed to save late layers model: {e}")
            # Fallback to state dict saving
            try:
                logger.info("Trying fallback to state dict saving")
                torch.save(model.state_dict(), os.path.join(late_path, "model_state_dict.pt"))
                # Save verification config separately
                with open(os.path.join(late_path, "verification_config.json"), "w") as f:
                    json.dump({
                        "adapter_locations": late_layers,
                        "hidden_size": model.hidden_size if hasattr(model, "hidden_size") else None,
                        "enable_cross_layer": model.enable_cross_layer if hasattr(model, "enable_cross_layer") else False,
                    }, f)
                logger.info(f"Late layers model state dict saved to {late_path}")
            except Exception as e2:
                logger.error(f"Failed to save state dict: {e2}")

    finally:
        # Restore original configuration
        model.verification_adapters = original_adapters
        model.adapter_locations = original_locations

    return True

def generate_ablation_models(verified_model_path, output_dir, tokenizer_path=None):
    """
    Generate all ablation variants of the verification model

    Args:
        verified_model_path: Path to the verification-enhanced model
        output_dir: Directory to save ablation variants
        tokenizer_path: Path to tokenizer (if different from model path)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load verification model
    logger.info(f"Loading verification model from {verified_model_path}")

    try:
        # Try to load with verification wrapper
        from enhanced_verification import load_bayesian_verification_model
        model = load_bayesian_verification_model(verified_model_path)
        logger.info("Loaded model with verification wrapper")
    except Exception as e:
        logger.warning(f"Error loading with verification wrapper: {e}")
        logger.warning("Trying standard loading method")

        # Fallback to standard loading
        model = AutoModelForCausalLM.from_pretrained(verified_model_path)

        # Check if it has verification attributes
        if not hasattr(model, 'verification_adapters'):
            logger.error("Model does not have verification components. Cannot create ablations.")
            return

    # Load tokenizer
    if tokenizer_path is None:
        tokenizer_path = verified_model_path

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # Save tokenizer with each ablation model
    except Exception as e:
        logger.warning(f"Error loading tokenizer: {e}")
        tokenizer = None

    # Create ablation variants
    ablation_paths = {}

    # 1. No cross-layer verification
    no_cross_path = os.path.join(output_dir, "no_cross_layer")
    os.makedirs(no_cross_path, exist_ok=True)
    if disable_cross_layer_verification(model, no_cross_path):
        ablation_paths["no_cross_layer"] = no_cross_path
        if tokenizer is not None:
            tokenizer.save_pretrained(no_cross_path)

    # 2. Fixed confidence thresholds
    fixed_conf_path = os.path.join(output_dir, "fixed_confidence")
    os.makedirs(fixed_conf_path, exist_ok=True)
    if fixed_confidence_thresholds(model, fixed_conf_path):
        ablation_paths["fixed_confidence"] = fixed_conf_path
        if tokenizer is not None:
            tokenizer.save_pretrained(fixed_conf_path)

    # 3. Single layer verification (middle layer)
    if hasattr(model, 'adapter_locations') and len(model.adapter_locations) > 0:
        # Choose middle layer
        middle_idx = len(model.adapter_locations) // 2
        middle_layer = model.adapter_locations[middle_idx]

        single_layer_path = os.path.join(output_dir, "single_layer")
        os.makedirs(single_layer_path, exist_ok=True)
        if single_layer_verification(model, single_layer_path, middle_layer):
            ablation_paths["single_layer"] = single_layer_path
            if tokenizer is not None:
                tokenizer.save_pretrained(single_layer_path)

    # 4. Early vs late layers
    depth_path = os.path.join(output_dir, "depth_variants")
    os.makedirs(depth_path, exist_ok=True)
    if early_late_verification(model, depth_path):
        ablation_paths["early_layers"] = os.path.join(depth_path, "early_layers")
        ablation_paths["late_layers"] = os.path.join(depth_path, "late_layers")
        if tokenizer is not None:
            tokenizer.save_pretrained(os.path.join(depth_path, "early_layers"))
            tokenizer.save_pretrained(os.path.join(depth_path, "late_layers"))

    # Save ablation paths index for easy reference
    with open(os.path.join(output_dir, "ablation_paths.json"), "w") as f:
        json.dump(ablation_paths, f, indent=2)

    logger.info(f"Created {len(ablation_paths)} ablation variants")
    for name, path in ablation_paths.items():
        logger.info(f"  {name}: {path}")

    return ablation_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ablation variants of a verification model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to verification model")
    parser.add_argument("--output_dir", type=str, default="ablation_models", help="Output directory")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer (if different)")

    args = parser.parse_args()

    generate_ablation_models(
        verified_model_path=args.model_path,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path
    )
