"""
Script to create parameter-matched baselines for fair comparison with 
verification-enhanced models.

This creates:
1. A standard adapter-only model (no verification mechanism)
2. A LoRA-enhanced model with similar parameter count
3. An MLP-adapter model (verification without cross-layer consistency)

These serve as controls to ensure improvements come from the verification
mechanism rather than just from additional parameters.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse
from peft import LoraConfig, get_peft_model

class SimpleAdapter(nn.Module):
    """
    Standard adapter with no verification mechanism - 
    used for parameter-matched baseline
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
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize with small weights for stable fine-tuning
        with torch.no_grad():
            nn.init.normal_(self.down_proj.weight, std=adapter_init_scale)
            nn.init.normal_(self.up_proj.weight, std=adapter_init_scale)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        # Down projection
        x = self.down_proj(hidden_states)
        x = torch.nn.functional.gelu(x)

        # Up projection
        x = self.up_proj(x)
        x = self.dropout(x)

        # Residual connection
        out = self.layer_norm(residual + x)

        return out


class MLPAdapter(nn.Module):
    """
    MLP-based adapter without verification confidence mechanism
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
        self.middle_layer = nn.Linear(bottleneck_size, bottleneck_size)
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize with small weights
        with torch.no_grad():
            nn.init.normal_(self.down_proj.weight, std=adapter_init_scale)
            nn.init.normal_(self.middle_layer.weight, std=adapter_init_scale)
            nn.init.normal_(self.up_proj.weight, std=adapter_init_scale)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.middle_layer.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        # Down projection
        x = self.down_proj(hidden_states)
        x = torch.nn.functional.gelu(x)

        # Middle layer
        x = self.middle_layer(x)
        x = torch.nn.functional.gelu(x)

        # Up projection
        x = self.up_proj(x)
        x = self.dropout(x)

        # Residual connection
        out = self.layer_norm(residual + x)

        return out


class AdapterWrapper(nn.Module):
    """
    Wrapper to add adapters to a pre-trained model
    """
    def __init__(
        self,
        base_model: nn.Module,
        adapter_locations: list,
        adapter_type: str = "simple",  # "simple" or "mlp"
        hidden_size: int = None,
        bottleneck_size: int = 64,
        freeze_base_model: bool = True
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
                raise ValueError("Cannot automatically determine hidden size")

        self.hidden_size = hidden_size
        self.adapter_locations = adapter_locations

        # Initialize adapters
        if adapter_type == "simple":
            self.adapters = nn.ModuleDict({
                f"layer_{layer_idx}": SimpleAdapter(
                    hidden_size=hidden_size,
                    bottleneck_size=bottleneck_size
                )
                for layer_idx in adapter_locations
            })
        elif adapter_type == "mlp":
            self.adapters = nn.ModuleDict({
                f"layer_{layer_idx}": MLPAdapter(
                    hidden_size=hidden_size,
                    bottleneck_size=bottleneck_size
                )
                for layer_idx in adapter_locations
            })
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

        # Freeze base model if specified
        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        # Get device
        device = next(self.parameters()).device

        # Ensure inputs are on the correct device
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        # Register hooks for each layer
        hooks = []

        def get_hook_fn(layer_idx):
            def hook_fn(module, input, output):
                if layer_idx in self.adapter_locations:
                    # Get the hidden states from output
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output

                    # Apply adapter
                    adapter = self.adapters[f"layer_{layer_idx}"]
                    modified_states = adapter(hidden_states)

                    # Replace output
                    if isinstance(output, tuple):
                        return (modified_states,) + output[1:]
                    else:
                        return modified_states

                return output
            return hook_fn

        # Get transformer layers
        layers = self._get_layers()

        # Register hooks
        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(get_hook_fn(i))
            hooks.append(hook)

        # Forward pass
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return outputs

    def _get_layers(self):
        """Get the transformer layers from the base model"""
        # Same layer detection logic as in the verification wrapper
        if hasattr(self.base_model, 'transformer'):
            if hasattr(self.base_model.transformer, 'h'):  # GPT-2 style
                return self.base_model.transformer.h
            elif hasattr(self.base_model.transformer, 'layer'):  # BERT style
                return self.base_model.transformer.layer
            elif hasattr(self.base_model.transformer, 'layers'):
                return self.base_model.transformer.layers
            elif hasattr(self.base_model.transformer, 'blocks'):
                return self.base_model.transformer.blocks

        elif hasattr(self.base_model, 'model'):
            if hasattr(self.base_model.model, 'decoder') and hasattr(self.base_model.model.decoder, 'layers'):
                return self.base_model.model.decoder.layers
            elif hasattr(self.base_model.model, 'encoder') and hasattr(self.base_model.model.encoder, 'layers'):
                return self.base_model.model.encoder.layers
            elif hasattr(self.base_model.model, 'layers'):
                return self.base_model.model.layers

        elif hasattr(self.base_model, 'layers'):
            return self.base_model.layers

        elif hasattr(self.base_model, 'blocks'):
            return self.base_model.blocks

        raise ValueError("Could not find transformer layers")

    def save_pretrained(self, output_dir):
        """Save the model"""
        os.makedirs(output_dir, exist_ok=True)

        # Save the base model
        self.base_model.save_pretrained(output_dir)

        # Save adapters
        torch.save(self.adapters.state_dict(), os.path.join(output_dir, "adapters.pt"))

        # Save adapter config
        config = {
            "adapter_locations": self.adapter_locations,
            "hidden_size": self.hidden_size,
        }

        import json
        with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
            json.dump(config, f)


def create_adapter_model(
    model_name: str,
    adapter_locations: list = None,
    adapter_type: str = "simple",
    bottleneck_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Create a model with standard adapters"""
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Determine number of layers if adapter_locations not specified
    if adapter_locations is None:
        num_layers = getattr(base_model.config, 'num_hidden_layers', 
                           getattr(base_model.config, 'n_layer', 
                                  getattr(base_model.config, 'num_layers', 12)))
        adapter_locations = list(range(2, num_layers, 3))

    # Create adapter wrapper
    model = AdapterWrapper(
        base_model=base_model,
        adapter_locations=adapter_locations,
        adapter_type=adapter_type,
        bottleneck_size=bottleneck_size,
        freeze_base_model=True
    )

    return model


def create_lora_model(
    model_name: str,
    r: int = 16,
    target_modules: list = None,
    alpha: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Create a model with LoRA adapters"""
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Determine target modules if not specified
    if target_modules is None:
        # Default to query and value projection in attention
        target_modules = ["q_proj", "v_proj"]

    # Create LoRA config
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Create LoRA model
    model = get_peft_model(base_model, lora_config)

    return model


def create_parameter_matched_baselines(
    model_name: str,
    verification_model_path: str,
    adapter_output_dir: str = "adapter_baseline",
    lora_output_dir: str = "lora_baseline",
    mlp_output_dir: str = "mlp_baseline",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Create parameter-matched baselines to compare with verification model

    Args:
        model_name: Base model name
        verification_model_path: Path to verification model (to match parameters)
        adapter_output_dir: Output directory for adapter model
        lora_output_dir: Output directory for LoRA model
        mlp_output_dir: Output directory for MLP adapter model
        device: Device to use
    """
    # Load verification model to get adapter locations and parameter count
    try:
        from enhanced_verification import load_bayesian_verification_model
        verification_model = load_bayesian_verification_model(verification_model_path)
        adapter_locations = verification_model.adapter_locations

        # Count trainable parameters in verification model
        verification_trainable = sum(p.numel() for p in verification_model.parameters() if p.requires_grad)
        print(f"Verification model trainable parameters: {verification_trainable:,}")

    except Exception as e:
        print(f"Error loading verification model: {e}")
        # Fallback to default adapter locations
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        num_layers = getattr(base_model.config, 'num_hidden_layers', 
                           getattr(base_model.config, 'n_layer', 
                                  getattr(base_model.config, 'num_layers', 12)))
        adapter_locations = list(range(2, num_layers, 3))
        verification_trainable = None

    # 1. Create adapter model
    adapter_model = create_adapter_model(
        model_name=model_name,
        adapter_locations=adapter_locations,
        adapter_type="simple",
        device=device
    )

    # Count trainable parameters
    adapter_trainable = sum(p.numel() for p in adapter_model.parameters() if p.requires_grad)
    print(f"Adapter model trainable parameters: {adapter_trainable:,}")

    # Save adapter model
    adapter_model.save_pretrained(adapter_output_dir)
    print(f"Adapter model saved to {adapter_output_dir}")

    # 2. Create MLP adapter model
    mlp_model = create_adapter_model(
        model_name=model_name,
        adapter_locations=adapter_locations,
        adapter_type="mlp",
        device=device
    )

    # Count trainable parameters
    mlp_trainable = sum(p.numel() for p in mlp_model.parameters() if p.requires_grad)
    print(f"MLP adapter model trainable parameters: {mlp_trainable:,}")

    # Save MLP adapter model
    mlp_model.save_pretrained(mlp_output_dir)
    print(f"MLP adapter model saved to {mlp_output_dir}")

    # 3. Create LoRA model with matched parameter count
    if verification_trainable:
        # Calculate LoRA rank to match parameter count
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        hidden_size = getattr(base_model.config, 'hidden_size', 
                            getattr(base_model.config, 'd_model', 
                                   getattr(base_model.config, 'n_embd', 768)))

        # Each LoRA adapter adds 2 * hidden_size * r parameters per target module
        # We target q_proj and v_proj by default
        target_modules = ["q_proj", "v_proj"]
        params_per_r = 2 * hidden_size * len(target_modules)

        # Calculate r to match verification model parameters
        r = min(256, max(8, int(verification_trainable / params_per_r)))

        print(f"Using LoRA rank r={r} to match parameter count")

        lora_model = create_lora_model(
            model_name=model_name,
            r=r,
            target_modules=target_modules,
            device=device
        )

        # Count trainable parameters
        lora_trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        print(f"LoRA model trainable parameters: {lora_trainable:,}")

        # Save LoRA model
        lora_model.save_pretrained(lora_output_dir)
        print(f"LoRA model saved to {lora_output_dir}")
    else:
        print("Skipping LoRA model creation (verification model parameter count unknown)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create parameter-matched baselines")
    parser.add_argument("--model_name", type=str, required=True, help="Base model name")
    parser.add_argument("--verification_model", type=str, required=True, help="Path to verification model")
    parser.add_argument("--adapter_output", type=str, default="adapter_baseline", help="Output dir for adapter model")
    parser.add_argument("--lora_output", type=str, default="lora_baseline", help="Output dir for LoRA model")
    parser.add_argument("--mlp_output", type=str, default="mlp_baseline", help="Output dir for MLP adapter model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")

    args = parser.parse_args()

    create_parameter_matched_baselines(
        model_name=args.model_name,
        verification_model_path=args.verification_model,
        adapter_output_dir=args.adapter_output,
        lora_output_dir=args.lora_output,
        mlp_output_dir=args.mlp_output,
        device=args.device
    )
