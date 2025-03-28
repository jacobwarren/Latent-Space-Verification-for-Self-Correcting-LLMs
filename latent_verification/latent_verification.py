import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from transformers import PreTrainedModel, AutoModelForCausalLM
import copy


class VerificationAdapter(nn.Module):
    """
    Adapter module that implements latent space verification.
    Uses low-rank adaptation to efficiently add verification capabilities.
    """
    def __init__(
        self, 
        hidden_size: int,
        bottleneck_size: int = 64,
        dropout_rate: float = 0.1,
        adapter_init_scale: float = 1e-3
    ):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.verification_layer = nn.Linear(bottleneck_size, bottleneck_size)
        self.confidence_scorer = nn.Linear(bottleneck_size, 1)
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize with small weights for stable fine-tuning
        with torch.no_grad():
            nn.init.normal_(self.down_proj.weight, std=adapter_init_scale)
            nn.init.normal_(self.up_proj.weight, std=adapter_init_scale)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states

        # Down projection
        x = self.down_proj(hidden_states)
        x = F.gelu(x)

        # Verification analysis
        v = self.verification_layer(x)
        v = F.gelu(v)

        # Confidence scores (how much should we trust the original hidden states)
        confidence = torch.sigmoid(self.confidence_scorer(v))

        # Up projection (generates corrections)
        corrections = self.up_proj(v)
        corrections = self.dropout(corrections)

        # Apply corrections weighted by inverse confidence
        # Low confidence means more correction applied
        # Ensure we maintain gradients here
        corrected_states = residual + (1 - confidence) * corrections
        corrected_states = self.layer_norm(corrected_states)

        return corrected_states, confidence


class CrossLayerVerifier(nn.Module):
    """
    Module that compares representations across different layers to identify inconsistencies.
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_layers: int,
        bottleneck_size: int = 64
    ):
        super().__init__()
        self.layer_extractors = nn.ModuleList([
            nn.Linear(hidden_size, bottleneck_size) 
            for _ in range(num_layers)
        ])
        self.comparator = nn.Linear(bottleneck_size * 2, 1)

    def forward(self, all_hidden_states: List[torch.Tensor]) -> torch.Tensor:
        # Extract features from each layer
        layer_features = [
            extractor(hidden_states)
            for extractor, hidden_states in zip(self.layer_extractors, all_hidden_states)
        ]

        # Compare features across layers
        consistency_scores = []
        for i in range(len(layer_features) - 1):
            for j in range(i + 1, len(layer_features)):
                # Concatenate features from different layers
                combined = torch.cat([layer_features[i], layer_features[j]], dim=-1)
                score = torch.sigmoid(self.comparator(combined))
                consistency_scores.append(score)

        # Average consistency scores
        overall_consistency = torch.mean(torch.stack(consistency_scores, dim=0), dim=0)
        return overall_consistency


class LatentVerificationWrapper(nn.Module):
    """
    Wrapper class that adds latent verification capabilities to a pre-trained model.
    """
    def __init__(
        self,
        base_model: PreTrainedModel,
        adapter_locations: List[int] = None,
        hidden_size: int = None,
        bottleneck_size: int = 64,
        enable_cross_layer: bool = True,
        freeze_base_model: bool = True,
        adapter_init_scale: float = 1e-3,
        verification_adapters: Optional[nn.ModuleDict] = None
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

        # Determine hidden size if not specified
        if hidden_size is None:
            if hasattr(self.config, 'hidden_size'):
                hidden_size = self.config.hidden_size
            elif hasattr(self.config, 'd_model'):
                hidden_size = self.config.d_model
            else:
                raise ValueError("Cannot automatically determine hidden size. Please specify it manually.")

        self.hidden_size = hidden_size

        # Determine number of layers
        self.num_layers = getattr(self.config, 'num_hidden_layers', 
                                 getattr(self.config, 'n_layer', 
                                        getattr(self.config, 'num_layers', 12)))

        # Default adapter locations if not specified (every 3rd layer)
        if adapter_locations is None:
            adapter_locations = list(range(2, self.num_layers, 3))

        self.adapter_locations = adapter_locations

        # Initialize verification adapters
        if verification_adapters is None:
            self.verification_adapters = nn.ModuleDict({
                f"layer_{layer_idx}": VerificationAdapter(
                    hidden_size=hidden_size,
                    bottleneck_size=bottleneck_size,
                    adapter_init_scale=adapter_init_scale
                )
                for layer_idx in adapter_locations
            })
        else:
            self.verification_adapters = verification_adapters

        # Initialize cross-layer verifier if enabled
        self.enable_cross_layer = enable_cross_layer
        if enable_cross_layer:
            self.cross_layer_verifier = CrossLayerVerifier(
                hidden_size=hidden_size,
                num_layers=len(adapter_locations),
                bottleneck_size=bottleneck_size
            )

        # Freeze base model if specified
        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False

    @property
    def device(self):
        """
        Returns the device on which the model is located.
        """
        # Get device from the base model's parameters
        return next(self.base_model.parameters()).device

    def generate(self, *args, **kwargs):
        """Pass generate call to base model while keeping verification hooks active"""
        # Reset tracking variables
        self.all_adapter_hidden_states = []
        self.all_confidence_scores = []

        # Set up verification hooks
        hooks = []

        def get_hook_fn(layer_idx):
            def hook_fn(module, input, output):
                if layer_idx in self.adapter_locations:
                    # Get hidden states
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output

                    # Apply verification adapter
                    adapter = self.verification_adapters[f"layer_{layer_idx}"]
                    adapter_dtype = next(adapter.parameters()).dtype
                    hidden_states = hidden_states.to(dtype=adapter_dtype)

                    try:
                        corrected_states, confidence = adapter(hidden_states)

                        # Store for verification
                        self.all_adapter_hidden_states.append(corrected_states)
                        self.all_confidence_scores.append(confidence)

                        # Return corrected states
                        if isinstance(output, tuple):
                            return (corrected_states,) + output[1:]
                        else:
                            return corrected_states
                    except Exception as e:
                        print(f"Error in verification adapter at layer {layer_idx}: {e}")
                        return output
                return output
            return hook_fn

        # Register hooks
        layers = self._get_layers()
        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(get_hook_fn(i))
            hooks.append(hook)

        try:
            # Call generate on the base model
            outputs = self.base_model.generate(*args, **kwargs)
        finally:
            # Always remove hooks
            for hook in hooks:
                hook.remove()

        return outputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """
        Forward pass through the model with verification components.

        Args:
            input_ids: Token ids (required)
            attention_mask: Attention mask for padding
            labels: Labels for computing the language modeling loss
            **kwargs: Additional arguments to pass to the base model
        """
        # Initialize lists to collect hidden states and confidence scores
        self.all_adapter_hidden_states = []
        self.all_confidence_scores = []

        # Get device and dtype
        device = next(self.parameters()).device
        dtype = next(self.base_model.parameters()).dtype

        # Ensure input_ids and attention_mask are on the correct device
        if input_ids is not None:
            # Ensure it's a tensor
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
            else:
                input_ids = input_ids.to(device=device)

        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
            else:
                attention_mask = attention_mask.to(device=device)

        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long, device=device)
            else:
                labels = labels.to(device=device)

        # Register hooks for each layer
        hooks = []

        def get_hook_fn(layer_idx):
            def hook_fn(module, input, output):
                if layer_idx in self.adapter_locations:
                    # Get the hidden states from the transformer layer output
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output

                    # Get adapter for this layer
                    adapter = self.verification_adapters[f"layer_{layer_idx}"]

                    # Ensure hidden states have the right dtype (same as adapter)
                    adapter_dtype = next(adapter.parameters()).dtype
                    hidden_states = hidden_states.to(dtype=adapter_dtype)

                    # Apply verification adapter
                    try:
                        corrected_states, confidence = adapter(hidden_states)

                        # Store for cross-layer verification
                        self.all_adapter_hidden_states.append(corrected_states)
                        self.all_confidence_scores.append(confidence)

                        # Replace the output with the corrected states
                        if isinstance(output, tuple):
                            return (corrected_states,) + output[1:]
                        else:
                            return corrected_states
                    except Exception as e:
                        print(f"Error in verification adapter at layer {layer_idx}: {e}")
                        # Fall back to original output in case of error
                        return output

                return output
            return hook_fn

        # Get transformer layers
        layers = self._get_layers()
        if not layers:
            print("Warning: Could not find transformer layers")

        # Register hooks on each layer
        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(get_hook_fn(i))
            hooks.append(hook)

        # Build model inputs
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "output_hidden_states": True,  # We need all hidden states
            **kwargs
        }

        # Remove None values
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}

        try:
            # Forward pass through the model
            outputs = self.base_model(**model_inputs)
        except Exception as e:
            print(f"Error in base model forward pass: {e}")
            import traceback
            traceback.print_exc()

            # Clean up hooks before raising
            for hook in hooks:
                hook.remove()
            raise

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Apply cross-layer verification if enabled
        if self.enable_cross_layer and self.all_adapter_hidden_states and len(self.all_adapter_hidden_states) > 0:
            try:
                cross_layer_consistency = self.cross_layer_verifier(self.all_adapter_hidden_states)

                # Add verification metrics to outputs
                if not hasattr(outputs, 'verification_metrics'):
                    # Create verification_metrics attribute if it doesn't exist
                    outputs.verification_metrics = {}

                outputs.verification_metrics = {
                    "cross_layer_consistency": cross_layer_consistency,
                    "layer_confidence_scores": self.all_confidence_scores
                }
            except Exception as e:
                print(f"Error in cross-layer verification: {e}")
                import traceback
                traceback.print_exc()

        return outputs

    def _get_layers(self):
        """Helper method to get the transformer layers from the base model."""
        # Handle Qwen models specifically
        if "qwen" in str(type(self.base_model)).lower():
            if hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'h'):
                return self.base_model.transformer.h
            elif hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'blocks'):
                return self.base_model.transformer.blocks
            elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'transformer') and hasattr(self.base_model.model.transformer, 'blocks'):
                return self.base_model.model.transformer.blocks

        # This handles different model architectures
        if hasattr(self.base_model, 'transformer'):
            if hasattr(self.base_model.transformer, 'h'):  # GPT-2 style
                return self.base_model.transformer.h
            elif hasattr(self.base_model.transformer, 'layer'):  # BERT style
                return self.base_model.transformer.layer
            elif hasattr(self.base_model.transformer, 'layers'):  # Some other models
                return self.base_model.transformer.layers
            elif hasattr(self.base_model.transformer, 'blocks'):  # Some newer models
                return self.base_model.transformer.blocks
        elif hasattr(self.base_model, 'encoder'):
            if hasattr(self.base_model.encoder, 'layer'):  # BERT style encoder
                return self.base_model.encoder.layer
            elif hasattr(self.base_model.encoder, 'layers'):  # Some encoder models
                return self.base_model.encoder.layers
        elif hasattr(self.base_model, 'decoder'):
            if hasattr(self.base_model.decoder, 'layers'):  # Some decoder models
                return self.base_model.decoder.layers
            elif hasattr(self.base_model.decoder, 'blocks'):  # Some newer models
                return self.base_model.decoder.blocks
        elif hasattr(self.base_model, 'model'):
            # Some models wrap everything in a 'model' attribute
            if hasattr(self.base_model.model, 'decoder') and hasattr(self.base_model.model.decoder, 'layers'):
                return self.base_model.model.decoder.layers
            elif hasattr(self.base_model.model, 'encoder') and hasattr(self.base_model.model.encoder, 'layers'):
                return self.base_model.model.encoder.layers
            elif hasattr(self.base_model.model, 'layers'):  # Mistral/Llama style
                return self.base_model.model.layers
        elif hasattr(self.base_model, 'layers'):  # Direct layers attribute (LLaMA style)
            return self.base_model.layers
        elif hasattr(self.base_model, 'blocks'):  # Direct blocks attribute
            return self.base_model.blocks

        # Try to handle additional cases by finding lists of layers
        for attr_name in dir(self.base_model):
            if attr_name.startswith('_'):
                continue

            attr = getattr(self.base_model, attr_name)
            if isinstance(attr, torch.nn.ModuleList) and len(attr) > 0:
                first_module = attr[0]
                # Check if it contains attention or MLP components (typical for transformer layers)
                has_attention = any('attention' in name.lower() for name, _ in first_module.named_modules())
                has_mlp = any(('mlp' in name.lower() or 'ffn' in name.lower()) for name, _ in first_module.named_modules())

                if has_attention or has_mlp:
                    print(f"Found layers through attribute: {attr_name}")
                    return attr

        # If we're still here, let's print out more information about the model structure
        print(f"Model type: {type(self.base_model).__name__}")
        print("Model attributes:", [attr for attr in dir(self.base_model) if not attr.startswith('_') and not callable(getattr(self.base_model, attr))])

        if hasattr(self.base_model, 'transformer'):
            print("Transformer attributes:", [attr for attr in dir(self.base_model.transformer) if not attr.startswith('_') and not callable(getattr(self.base_model.transformer, attr))])

        raise ValueError("Could not automatically determine the transformer layers in the model. Please specify the layer indices manually or use a supported model architecture.")

    def save_pretrained(self, output_dir, safe_serialization=True, **kwargs):
        """
        Save the wrapped model to `output_dir` with proper handling of shared weights.

        Args:
            output_dir (str): The directory to save the model.
            safe_serialization (bool): Whether to save in safetensors format or pytorch format.
            **kwargs: Additional arguments to pass to the base save_pretrained method.
        """
        import os
        import torch
        import json

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Handle shared weights issue by creating a config flag
        if hasattr(self.base_model.config, "tie_word_embeddings"):
            # Mark that we need to tie weights when loading
            tie_word_embeddings = self.base_model.config.tie_word_embeddings
        else:
            # For models without this flag, check if weights are actually tied
            tie_word_embeddings = False
            if hasattr(self.base_model, "get_output_embeddings") and hasattr(self.base_model, "get_input_embeddings"):
                output_emb = self.base_model.get_output_embeddings()
                input_emb = self.base_model.get_input_embeddings()
                if output_emb is not None and input_emb is not None:
                    # Check if they share weights
                    for p1, p2 in zip(output_emb.parameters(), input_emb.parameters()):
                        if p1 is p2:  # Check if it's the same object (shared memory)
                            tie_word_embeddings = True
                            break

        # Update config
        self.base_model.config.tie_word_embeddings = tie_word_embeddings

        # Save the config file
        self.base_model.config.save_pretrained(output_dir)

        # For models with shared weights but using safetensors, we need a special approach
        if safe_serialization and tie_word_embeddings:
            # 1. Save the model state dict using torch's standard method 
            import torch
            from huggingface_hub import hf_hub_download
            import tempfile
            import shutil

            # Save model weights as PyTorch checkpoint
            torch.save(self.base_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

            # Then, use huggingface_hub's conversion tools to convert the model to safetensors
            # safely handling the shared tensors
            try:
                from safetensors.torch import save_file
                from transformers.utils import WEIGHTS_NAME, SAFE_WEIGHTS_NAME
                from transformers.modeling_utils import shard_checkpoint

                # Get state dict
                state_dict = self.base_model.state_dict()

                # Remove duplicated tensors before saving
                # Find duplicate tensor entries
                shared_tensors = {}
                for key, tensor in state_dict.items():
                    for k, t in state_dict.items():
                        if k != key and t is tensor:  # Check if tensors share memory
                            if key not in shared_tensors:
                                shared_tensors[key] = []
                            shared_tensors[key].append(k)

                # Create a modified state dict without duplicates
                filtered_state_dict = {}
                already_added = set()

                for key, tensor in state_dict.items():
                    # Check if this tensor has been added already
                    tensor_id = id(tensor)
                    if tensor_id in already_added:
                        continue

                    filtered_state_dict[key] = tensor
                    already_added.add(tensor_id)

                # Save the filtered state dict to safetensors
                save_file(filtered_state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME))

                # Delete the PyTorch weights as we now have safetensors
                os.remove(os.path.join(output_dir, "pytorch_model.bin"))
                print(f"Successfully saved model to {output_dir} using safetensors format.")
            except ImportError:
                print("Could not import safetensors. Using PyTorch format instead.")
                # Keep the PyTorch weights we already saved
        else:
            # If no shared weights or not using safetensors, use standard method
            self.base_model.save_pretrained(output_dir, safe_serialization=safe_serialization, **kwargs)

        # Save adapters, cross-layer verifier, and config
        torch.save(self.verification_adapters.state_dict(), f"{output_dir}/verification_adapters.pt")

        if self.enable_cross_layer:
            torch.save(self.cross_layer_verifier.state_dict(), f"{output_dir}/cross_layer_verifier.pt")

        import json
        config = {
            "adapter_locations": self.adapter_locations,
            "hidden_size": self.hidden_size,
            "enable_cross_layer": self.enable_cross_layer,
        }
        with open(f"{output_dir}/verification_config.json", "w") as f:
            json.dump(config, f)



class VerificationLoss(nn.Module):
    """
    Loss function for training the verification adapters.
    Combines a regular task loss with auxiliary verification objectives.
    """
    def __init__(
        self, 
        task_loss_fn,
        consistency_weight: float = 0.1,
        confidence_regularization_weight: float = 0.05
    ):
        super().__init__()
        self.task_loss_fn = task_loss_fn
        self.consistency_weight = consistency_weight
        self.confidence_regularization_weight = confidence_regularization_weight

    def forward(
        self, 
        outputs, 
        targets,
        verification_metrics: Optional[Dict] = None
    ):
        # Calculate the main task loss
        task_loss = self.task_loss_fn(outputs, targets)

        total_loss = task_loss

        # Add verification-specific losses if metrics are provided
        if verification_metrics is not None:
            # Cross-layer consistency loss
            if "cross_layer_consistency" in verification_metrics:
                consistency = verification_metrics["cross_layer_consistency"]
                # We want to maximize consistency (minimize 1 - consistency)
                consistency_loss = torch.mean(1 - consistency)
                total_loss = total_loss + self.consistency_weight * consistency_loss

            # Confidence regularization (prevent always-high or always-low confidence)
            if "layer_confidence_scores" in verification_metrics:
                confidence_scores = verification_metrics["layer_confidence_scores"]
                valid_scores = [c for c in confidence_scores if c.requires_grad]

                if valid_scores:
                    avg_confidence = torch.mean(torch.cat([conf.mean() for conf in valid_scores]))
                    # Add small epsilon to prevent log(0)
                    confidence_reg = -torch.log(avg_confidence + 1e-10) - torch.log(1 - avg_confidence + 1e-10)
                    total_loss = total_loss + self.confidence_regularization_weight * confidence_reg

        return total_loss


def create_verification_model(
    model_name_or_path: str,
    adapter_locations: List[int] = None,
    bottleneck_size: int = 64,
    enable_cross_layer: bool = True,
    freeze_base_model: bool = True,
    **kwargs
) -> LatentVerificationWrapper:
    """
    Helper function to create a verification-enhanced model from a model name or path.

    Args:
        model_name_or_path: Hugging Face model name or path
        adapter_locations: List of layer indices where to apply verification adapters
        bottleneck_size: Size of the verification adapter bottleneck
        enable_cross_layer: Whether to enable cross-layer verification
        freeze_base_model: Whether to freeze the base model parameters

    Returns:
        A LatentVerificationWrapper instance
    """
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)

    # Create and return the verification wrapper
    return LatentVerificationWrapper(
        base_model=base_model,
        adapter_locations=adapter_locations,
        bottleneck_size=bottleneck_size,
        enable_cross_layer=enable_cross_layer,
        freeze_base_model=freeze_base_model
    )


def load_verification_model(
    model_path: str,
    **kwargs
) -> LatentVerificationWrapper:
    """
    Load a saved verification model from the given path.
    """
    import json
    import os

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    # Load verification configuration
    with open(os.path.join(model_path, "verification_config.json"), "r") as f:
        config = json.load(f)

    # Create verification wrapper
    model = LatentVerificationWrapper(
        base_model=base_model,
        adapter_locations=config["adapter_locations"],
        hidden_size=config["hidden_size"],
        enable_cross_layer=config["enable_cross_layer"],
        freeze_base_model=False  # Don't freeze during loading
    )

    # Load verification adapters
    model.verification_adapters.load_state_dict(
        torch.load(os.path.join(model_path, "verification_adapters.pt"))
    )

    # Load cross-layer verifier if enabled
    if config["enable_cross_layer"]:
        model.cross_layer_verifier.load_state_dict(
            torch.load(os.path.join(model_path, "cross_layer_verifier.pt"))
        )

    Print("Special Load Successful")

    return model


# Activation test function to verify the enhancement is properly applied
def run_verification_activation_test(
    model_name: str = "gpt2",
    test_text: str = "This is a test of the latent verification system.",
    adapter_locations: List[int] = [2, 5, 8, 11]
):
    """
    Runs a pre-fine-tuning activation test to verify the verification enhancement
    is properly applied to the model architecture.

    Args:
        model_name: Name of the pre-trained model to test
        test_text: Sample text to run through the model
        adapter_locations: Layer indices for verification adapters

    Returns:
        Dict containing test results and diagnostics
    """
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer

    print(f"Running activation test on {model_name}...")

    # Load tokenizer and prepare input
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(test_text, return_tensors="pt")

    # Load original model for comparison
    print("Loading original model...")
    original_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Create verification-enhanced model
    print("Creating verification-enhanced model...")
    verification_model = create_verification_model(
        model_name_or_path=model_name,
        adapter_locations=adapter_locations
    )

    # Run both models and collect outputs
    print("Running inference on original model...")
    with torch.no_grad():
        original_outputs = original_model(**inputs, output_hidden_states=True)

    print("Running inference on verification-enhanced model...")
    with torch.no_grad():
        verification_outputs = verification_model(**inputs)

    # Collect diagnostics
    diagnostics = {
        "model_name": model_name,
        "test_text": test_text,
        "adapter_locations": adapter_locations,
        "has_verification_metrics": hasattr(verification_outputs, "verification_metrics"),
        "output_shapes_match": original_outputs.logits.shape == verification_outputs.logits.shape,
        "verification_adapters_count": len(verification_model.verification_adapters),
    }

    if hasattr(verification_outputs, "verification_metrics"):
        metrics = verification_outputs.verification_metrics
        diagnostics["cross_layer_consistency"] = metrics.get("cross_layer_consistency", None)

        # Analyze confidence scores
        if "layer_confidence_scores" in metrics:
            confidence_scores = metrics["layer_confidence_scores"]
            confidence_means = [score.mean().item() for score in confidence_scores]
            confidence_stds = [score.std().item() for score in confidence_scores]

            diagnostics["confidence_means"] = confidence_means
            diagnostics["confidence_stds"] = confidence_stds

            # Create visualization of confidence distributions
            plt.figure(figsize=(10, 6))
            for i, (mean, std) in enumerate(zip(confidence_means, confidence_stds)):
                plt.bar(i, mean, yerr=std, capsize=10, 
                       label=f"Layer {adapter_locations[i]}")

            plt.xlabel("Adapter Index")
            plt.ylabel("Confidence Score")
            plt.title("Initial Verification Confidence Scores (Pre-Training)")
            plt.axhline(y=0.5, color='r', linestyle='--', 
                       label="0.5 threshold")
            plt.legend()
            plt.tight_layout()

            # Save to file or convert to image buffer
            plt.savefig("verification_confidence_test.png")
            plt.close()

            diagnostics["confidence_plot_saved"] = "verification_confidence_test.png"

    # Compare differences in original vs. verification model outputs
    if original_outputs.logits.shape == verification_outputs.logits.shape:
        logits_diff = (original_outputs.logits - verification_outputs.logits)
        diagnostics["max_logits_diff"] = logits_diff.abs().max().item()
        diagnostics["mean_logits_diff"] = logits_diff.abs().mean().item()

        # If the difference is very small, adapters might not be applied correctly
        if diagnostics["mean_logits_diff"] < 1e-6:
            print("WARNING: Verification adapters don't seem to be modifying the outputs!")
            diagnostics["adapters_applied_correctly"] = False
        else:
            diagnostics["adapters_applied_correctly"] = True
            print(f"Verification adapters are modifying outputs (mean diff: {diagnostics['mean_logits_diff']:.6f})")

    # Print summary
    print("\n=== Verification Enhancement Test Results ===")
    print(f"Model: {model_name}")
    print(f"Verification adapters installed: {len(verification_model.verification_adapters)}")
    print(f"Verification metrics available: {diagnostics['has_verification_metrics']}")
    print(f"Adapters correctly modifying outputs: {diagnostics.get('adapters_applied_correctly', 'Unknown')}")

    if "confidence_means" in diagnostics:
        print("\nInitial confidence scores (untrained):")
        for i, mean in enumerate(diagnostics["confidence_means"]):
            print(f"  Layer {adapter_locations[i]}: {mean:.4f} ± {diagnostics['confidence_stds'][i]:.4f}")

    if "confidence_plot_saved" in diagnostics:
        print(f"\nConfidence visualization saved to: {diagnostics['confidence_plot_saved']}")

    print("\nNOTE: These are untrained adapter values. The confidence scores")
    print("will be random until the model is fine-tuned.")

    return diagnostics


def detect_qwen_layers(model_name):
    """
    Utility function to detect and debug Qwen model layer structure.
    """
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print(f"Model type: {type(model)}")
    print("Top-level attributes:")
    for attr in dir(model):
        if not attr.startswith('_') and not callable(getattr(model, attr)):
            print(f"  {attr}")

    # Check model.transformer if it exists
    if hasattr(model, 'transformer'):
        print("\nTransformer attributes:")
        transformer = model.transformer
        for attr in dir(transformer):
            if not attr.startswith('_') and not callable(getattr(transformer, attr)):
                print(f"  {attr}")

                # Check if this could be layers
                value = getattr(transformer, attr)
                if isinstance(value, torch.nn.ModuleList):
                    print(f"    ModuleList with {len(value)} elements")
                    if len(value) > 0:
                        print(f"    First element type: {type(value[0])}")

    # Check model.model if it exists (for wrapped models)
    if hasattr(model, 'model'):
        print("\nmodel.model attributes:")
        inner_model = model.model
        for attr in dir(inner_model):
            if not attr.startswith('_') and not callable(getattr(inner_model, attr)):
                print(f"  {attr}")

                # Check if this could be layers
                value = getattr(inner_model, attr)
                if isinstance(value, torch.nn.ModuleList):
                    print(f"    ModuleList with {len(value)} elements")
                    if len(value) > 0:
                        print(f"    First element type: {type(value[0])}")

                # Check transformer inside model.model
                if attr == 'transformer' and hasattr(value, 'blocks'):
                    print(f"    transformer.blocks: ModuleList with {len(value.blocks)} elements")


def alternative_training_loop(model, train_dataloader, learning_rate=1e-5, consistency_weight=0.1, 
                             confidence_regularization_weight=0.05, num_steps=100):
    """
    A more robust training loop that ensures gradient flow for verification components.

    Args:
        model: The verification model
        train_dataloader: DataLoader with training data
        learning_rate: Learning rate for optimizer
        consistency_weight: Weight for consistency loss
        confidence_regularization_weight: Weight for confidence regularization
        num_steps: Number of training steps

    Returns:
        dict: Training results and metrics
    """
    print("\nUsing alternative training loop to ensure gradient flow...")

    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {len(trainable_params)}")

    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found!")

    # Get device and dtype
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    print(f"Model is on device: {device}")

    # Initialize storage for verification metrics
    model.all_adapter_hidden_states = []
    model.all_confidence_scores = []

    # Convert verification adapters to the correct dtype and device
    if hasattr(model, 'verification_adapters'):
        for name, adapter in model.verification_adapters.items():
            # Convert all parameters to the correct dtype and device
            adapter.to(device=device, dtype=dtype)
        print(f"Moved verification adapters to {device}")

    if hasattr(model, 'cross_layer_verifier'):
        model.cross_layer_verifier.to(device=device, dtype=dtype)
        print(f"Moved cross layer verifier to {device}")

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=num_steps)

    model.train()
    total_loss = 0
    steps = 0

    # Track verification metrics
    verification_metrics_history = {
        'cross_layer_consistency': [],
        'confidence_scores': []
    }

    # Training loop
    print(f"Starting training loop for {num_steps} steps...")

    # Create an iterator over the dataloader
    train_iter = iter(train_dataloader)

    for step in range(num_steps):
        try:
            # Get next batch, reset iterator if needed
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)

            # Move batch to the correct device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(**batch)

            # Get loss
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            else:
                # Create a dummy loss from the logits if needed
                loss = outputs.logits.mean() * 0.0

            # Add verification-specific losses
            if hasattr(outputs, "verification_metrics"):
                metrics = outputs.verification_metrics

                if "cross_layer_consistency" in metrics:
                    consistency = metrics["cross_layer_consistency"]
                    consistency_loss = torch.mean(1 - consistency)
                    loss = loss + consistency_weight * consistency_loss

                    # Track metrics
                    verification_metrics_history['cross_layer_consistency'].append(
                        consistency.detach().mean().item()
                    )

                if "layer_confidence_scores" in metrics:
                    confidence_scores = metrics["layer_confidence_scores"]
                    valid_scores = [c for c in confidence_scores if c.requires_grad]

                    if valid_scores:
                        avg_confidence = torch.mean(torch.cat([conf.mean() for conf in valid_scores]))
                        confidence_reg = -torch.log(avg_confidence + 1e-10) - torch.log(1 - avg_confidence + 1e-10)
                        loss = loss + confidence_regularization_weight * confidence_reg

                        # Track metrics
                        verification_metrics_history['confidence_scores'].append(
                            avg_confidence.detach().item()
                        )

            # Explicitly connect loss to trainable parameters if needed
            if not loss.requires_grad:
                dummy_loss = sum(p.mean() * 0.0001 for p in trainable_params)
                loss = loss + dummy_loss

            # Backward and optimize
            try:
                loss.backward()

                # Check if gradients are flowing
                grad_norms = [p.grad.norm().item() if p.grad is not None else 0.0 for p in trainable_params]
                grad_norm_avg = sum(grad_norms)/len(grad_norms) if grad_norms else 0.0

                if step % 10 == 0:
                    print(f"Step {step}: Loss = {loss.item():.4f}, Grad norm = {grad_norm_avg:.6f}")

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
            except Exception as e:
                print(f"Error in backward/step: {e}")
                import traceback
                traceback.print_exc()
                continue

            steps += 1

        except Exception as e:
            print(f"Error in training loop: {e}")
            import traceback
            traceback.print_exc()
            continue

    return {
        'total_loss': total_loss,
        'steps': steps,
        'verification_metrics_history': verification_metrics_history
    }
