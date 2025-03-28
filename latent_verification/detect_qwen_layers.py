"""
Utility script for detecting the architecture and layers of Qwen models.
This addresses the common issue where Qwen models have different internal
structures compared to other HuggingFace models.
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig
import argparse


def detect_qwen_layers(model_name_or_path, verbose=True):
    """
    Analyze the architecture of a Qwen model to identify its layers.
    
    Args:
        model_name_or_path: Name or path of the Qwen model
        verbose: Whether to print detailed information
        
    Returns:
        dict: Dictionary containing architectural information
    """
    if verbose:
        print(f"Analyzing Qwen model architecture: {model_name_or_path}")
    
    # Get config first to avoid loading full model if possible
    try:
        config = AutoConfig.from_pretrained(model_name_or_path)
        if verbose:
            print(f"Model config type: {type(config).__name__}")
            print(f"Hidden size: {getattr(config, 'hidden_size', None)}")
            print(f"Number of layers: {getattr(config, 'num_hidden_layers', None)}")
    except Exception as e:
        if verbose:
            print(f"Error loading config: {e}")
        config = None
    
    # Load actual model
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
    except Exception as e:
        print(f"Error loading model: {e}")
        try:
            print("Trying again with trust_remote_code=True...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, 
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load model: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # Collect model information
    model_info = {
        "model_type": type(model).__name__,
        "success": True,
        "hidden_size": None,
        "num_layers": None,
        "layer_locations": [],
        "layer_type": None,
        "layer_attributes": []
    }
    
    if verbose:
        print(f"\nModel type: {model_info['model_type']}")
        print("Top-level attributes:")
        for attr in dir(model):
            if not attr.startswith('_') and not callable(getattr(model, attr)):
                print(f"  {attr}")
    
    # Check for transformer attribute
    if hasattr(model, 'transformer'):
        transformer = model.transformer
        model_info["transformer_type"] = type(transformer).__name__
        
        if verbose:
            print("\nTransformer attributes:")
            for attr in dir(transformer):
                if not attr.startswith('_') and not callable(getattr(transformer, attr)):
                    print(f"  {attr}")
        
        # Check for blocks in transformer
        if hasattr(transformer, 'blocks'):
            blocks = transformer.blocks
            model_info["layer_type"] = "transformer.blocks"
            model_info["num_layers"] = len(blocks)
            model_info["layer_locations"] = list(range(len(blocks)))
            
            # Get hidden size
            if hasattr(blocks[0], 'ln_1'):
                model_info["hidden_size"] = blocks[0].ln_1.weight.shape[0]
            
            if verbose:
                print(f"\nFound {len(blocks)} transformer blocks")
                print(f"First block type: {type(blocks[0]).__name__}")
                print("Block attributes:")
                for attr in dir(blocks[0]):
                    if not attr.startswith('_') and not callable(getattr(blocks[0], attr)):
                        print(f"  {attr}")
                
                # Print attention type
                if hasattr(blocks[0], 'attn'):
                    print(f"Attention type: {type(blocks[0].attn).__name__}")
        
        # Check for h in transformer (GPT-style)
        elif hasattr(transformer, 'h'):
            layers = transformer.h
            model_info["layer_type"] = "transformer.h"
            model_info["num_layers"] = len(layers)
            model_info["layer_locations"] = list(range(len(layers)))
            
            # Get hidden size
            if hasattr(layers[0], 'ln_1'):
                model_info["hidden_size"] = layers[0].ln_1.weight.shape[0]
            
            if verbose:
                print(f"\nFound {len(layers)} transformer layers (h)")
                print(f"First layer type: {type(layers[0]).__name__}")
    
    # Check for model.model pattern
    elif hasattr(model, 'model'):
        inner_model = model.model
        model_info["inner_model_type"] = type(inner_model).__name__
        
        if verbose:
            print("\nmodel.model attributes:")
            for attr in dir(inner_model):
                if not attr.startswith('_') and not callable(getattr(inner_model, attr)):
                    print(f"  {attr}")
        
        # Check if model.model has transformer
        if hasattr(inner_model, 'transformer'):
            inner_transformer = inner_model.transformer
            
            if verbose:
                print("\nmodel.model.transformer attributes:")
                for attr in dir(inner_transformer):
                    if not attr.startswith('_') and not callable(getattr(inner_transformer, attr)):
                        print(f"  {attr}")
            
            # Check for blocks in inner transformer
            if hasattr(inner_transformer, 'blocks'):
                blocks = inner_transformer.blocks
                model_info["layer_type"] = "model.transformer.blocks"
                model_info["num_layers"] = len(blocks)
                model_info["layer_locations"] = list(range(len(blocks)))
                
                # Get hidden size
                if hasattr(blocks[0], 'ln_1'):
                    model_info["hidden_size"] = blocks[0].ln_1.weight.shape[0]
                
                if verbose:
                    print(f"\nFound {len(blocks)} transformer blocks in model.transformer")
                    print(f"First block type: {type(blocks[0]).__name__}")
        
        # Check for layers directly in model.model
        elif hasattr(inner_model, 'layers'):
            layers = inner_model.layers
            model_info["layer_type"] = "model.layers"
            model_info["num_layers"] = len(layers)
            model_info["layer_locations"] = list(range(len(layers)))
            
            # Get hidden size from first layer
            if len(layers) > 0 and hasattr(layers[0], 'input_layernorm'):
                model_info["hidden_size"] = layers[0].input_layernorm.weight.shape[0]
            
            if verbose:
                print(f"\nFound {len(layers)} layers in model.layers")
                print(f"First layer type: {type(layers[0]).__name__}")
    
    # Recommend adapter locations
    if model_info["num_layers"]:
        # Default to every third layer, plus first and last quarters
        model_info["recommended_adapter_locations"] = []
        num_layers = model_info["num_layers"]
        
        # Include layers near the beginning, middle, and end
        model_info["recommended_adapter_locations"] = [
            # First quarter
            num_layers // 4,
            # Middle layers
            num_layers // 2 - 1,
            num_layers // 2 + 1,
            # Last quarter
            3 * num_layers // 4,
        ]
        
        if verbose:
            print(f"\nRecommended adapter locations: {model_info['recommended_adapter_locations']}")
    
    # Summary
    if verbose:
        print("\n=== Model Architecture Summary ===")
        print(f"Model type: {model_info['model_type']}")
        print(f"Number of layers: {model_info['num_layers']}")
        print(f"Hidden size: {model_info['hidden_size']}")
        print(f"Layer type: {model_info['layer_type']}")
        print(f"Recommended adapter locations: {model_info.get('recommended_adapter_locations', [])}")
    
    return model_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect layer architecture in Qwen models")
    parser.add_argument("model_name", type=str, help="Model name or path")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    
    args = parser.parse_args()
    
    detect_qwen_layers(args.model_name, verbose=not args.quiet)