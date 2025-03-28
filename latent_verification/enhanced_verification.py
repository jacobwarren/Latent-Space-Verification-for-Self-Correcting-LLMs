import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

class BayesianVerificationAdapter(nn.Module):
    """
    Enhanced verification adapter that implements uncertainty-weighted corrections
    using Bayesian principles to model epistemic uncertainty.
    """
    def __init__(
        self, 
        hidden_size: int,
        bottleneck_size: int = 64,
        dropout_rate: float = 0.1,
        adapter_init_scale: float = 1e-3,
        num_monte_carlo_samples: int = 5
    ):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.verification_layer = nn.Linear(bottleneck_size, bottleneck_size)

        # Instead of point estimate, output mean and log variance for confidence
        self.confidence_mean = nn.Linear(bottleneck_size, 1)
        self.confidence_logvar = nn.Linear(bottleneck_size, 1)

        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.num_monte_carlo_samples = num_monte_carlo_samples

        # Initialize with small weights for stable fine-tuning
        with torch.no_grad():
            nn.init.normal_(self.down_proj.weight, std=adapter_init_scale)
            nn.init.normal_(self.up_proj.weight, std=adapter_init_scale)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure hidden states and adapter parameters have the same dtype
        device = self.down_proj.weight.device
        dtype = self.down_proj.weight.dtype

        # Convert hidden states to the same dtype as the adapter
        if hidden_states.dtype != dtype:
            hidden_states = hidden_states.to(dtype=dtype)

        # Save original for residual connection
        residual = hidden_states

        # Down projection
        x = self.down_proj(hidden_states)
        x = F.gelu(x)

        # Verification analysis
        v = self.verification_layer(x)
        v = F.gelu(v)

        # Confidence distribution parameters
        mean = torch.sigmoid(self.confidence_mean(v))
        logvar = self.confidence_logvar(v)

        # Monte Carlo sampling for Bayesian estimation
        if self.training and self.num_monte_carlo_samples > 1:
            # Sample multiple confidence values to model uncertainty
            samples = []
            for _ in range(self.num_monte_carlo_samples):
                # Sample from the distribution
                epsilon = torch.randn_like(logvar, device=device, dtype=dtype)
                sample = mean + torch.exp(0.5 * logvar) * epsilon
                sample = torch.sigmoid(sample)  # Constrain to [0,1]
                samples.append(sample)

            # Use the mean of samples as confidence
            confidence = torch.mean(torch.stack(samples, dim=0), dim=0)
            # Calculate uncertainty as variance across samples
            uncertainty = torch.var(torch.stack(samples, dim=0), dim=0)
        else:
            # During inference, just use the mean
            confidence = mean
            # Uncertainty derived from logvar
            uncertainty = torch.exp(logvar)

        # Up projection (generates corrections)
        corrections = self.up_proj(v)
        corrections = self.dropout(corrections)

        # Apply corrections weighted by inverse confidence
        # Low confidence means more correction applied
        # High uncertainty means apply less correction (more cautious)
        correction_weight = (1 - confidence) * torch.exp(-uncertainty)
        corrected_states = residual + correction_weight * corrections
        corrected_states = self.layer_norm(corrected_states)

        return corrected_states, confidence


class HierarchicalVerifier(nn.Module):
    """
    Implements verification at multiple abstraction levels:
    - Token level: Focused on individual token representations
    - Phrase level: Considers local context across multiple tokens
    - Semantic level: Analyzes overall meaning and consistency
    """
    def __init__(
        self, 
        hidden_size: int,
        bottleneck_size: int = 64,
        num_heads: int = 4,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        # Token-level verification
        self.token_verifier = nn.Sequential(
            nn.Linear(hidden_size, bottleneck_size),
            nn.GELU(),
            nn.Linear(bottleneck_size, bottleneck_size),
            nn.Dropout(dropout_rate)
        )

        # Phrase-level verification using attention
        self.phrase_attention = nn.MultiheadAttention(
            embed_dim=bottleneck_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        # Semantic-level verification
        self.semantic_verifier = nn.Sequential(
            nn.Linear(bottleneck_size, bottleneck_size),
            nn.GELU(),
            nn.Linear(bottleneck_size, bottleneck_size),
            nn.Dropout(dropout_rate)
        )

        # Output projection
        self.output_proj = nn.Linear(bottleneck_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Confidence scorers at each level
        self.token_confidence = nn.Linear(bottleneck_size, 1)
        self.phrase_confidence = nn.Linear(bottleneck_size, 1)
        self.semantic_confidence = nn.Linear(bottleneck_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        residual = hidden_states
        batch_size, seq_len, _ = hidden_states.shape

        # Token-level verification
        token_features = self.token_verifier(hidden_states)
        token_conf = torch.sigmoid(self.token_confidence(token_features))

        # Phrase-level verification (attention-based)
        phrase_features, _ = self.phrase_attention(
            token_features, token_features, token_features
        )
        phrase_conf = torch.sigmoid(self.phrase_confidence(phrase_features))

        # Semantic-level verification
        semantic_features = self.semantic_verifier(phrase_features)
        semantic_conf = torch.sigmoid(self.semantic_confidence(semantic_features))

        # Final verified representation
        verified_features = self.output_proj(semantic_features)
        verified_states = self.layer_norm(residual + verified_features)

        # Collect confidence metrics
        confidences = {
            "token_level": token_conf,
            "phrase_level": phrase_conf,
            "semantic_level": semantic_conf,
            # Combined confidence (weighted product)
            "combined": token_conf * 0.2 + phrase_conf * 0.3 + semantic_conf * 0.5
        }

        return verified_states, confidences["combined"]


class ResidualVerificationNetwork(nn.Module):
    """
    Implements a verification module with stronger residual connections
    to preserve important information while allowing targeted corrections.
    """
    def __init__(
        self, 
        hidden_size: int,
        bottleneck_size: int = 64,
        num_experts: int = 4,
        dropout_rate: float = 0.1,
        adapter_init_scale: float = 1e-3
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Mixture of Expert verifiers for different aspects of the content
        self.expert_routers = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_experts)
        ])

        self.expert_verifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, bottleneck_size),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(bottleneck_size, bottleneck_size)
            ) for _ in range(num_experts)
        ])

        # Output projections for each expert
        self.expert_outputs = nn.ModuleList([
            nn.Linear(bottleneck_size, hidden_size) for _ in range(num_experts)
        ])

        # Confidence scorers for each expert
        self.expert_confidences = nn.ModuleList([
            nn.Linear(bottleneck_size, 1) for _ in range(num_experts)
        ])

        # Gating mechanism to control information flow from original to corrected
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        # Initialize with small weights
        for i in range(num_experts):
            with torch.no_grad():
                nn.init.normal_(self.expert_outputs[i].weight, std=adapter_init_scale)
                nn.init.zeros_(self.expert_outputs[i].bias)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        normalized = self.layer_norm(hidden_states)

        # Calculate routing weights
        routing_weights = torch.cat([
            router(normalized) for router in self.expert_routers
        ], dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)

        # Apply each expert
        expert_outputs = []
        expert_confidences = []

        for i, (verifier, output_proj, conf_proj) in enumerate(
            zip(self.expert_verifiers, self.expert_outputs, self.expert_confidences)
        ):
            # Process through expert
            features = verifier(normalized)
            expert_output = output_proj(features)

            # Calculate confidence
            confidence = torch.sigmoid(conf_proj(features))

            # Weight by router
            expert_weight = routing_weights[..., i:i+1]
            weighted_output = expert_output * expert_weight
            weighted_confidence = confidence * expert_weight

            expert_outputs.append(weighted_output)
            expert_confidences.append(weighted_confidence)

        # Combine expert outputs
        combined_output = sum(expert_outputs)
        combined_confidence = sum(expert_confidences)

        # Apply adaptive residual gate
        gate_input = torch.cat([residual, combined_output], dim=-1)
        gate_value = self.gate(gate_input)

        # Final output with gated residual
        corrected_states = (gate_value * residual) + ((1 - gate_value) * combined_output)

        return corrected_states, combined_confidence


class KnowledgeGroundedVerifier(nn.Module):
    """
    Verification module that incorporates external knowledge to ground verification.
    This version simulates knowledge grounding with a pre-trained frozen factual encoder.
    """
    def __init__(
        self,
        hidden_size: int,
        knowledge_size: int,
        bottleneck_size: int = 64,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.knowledge_size = knowledge_size

        # Project model hidden states to knowledge space
        self.hidden_to_knowledge = nn.Linear(hidden_size, knowledge_size)

        # Knowledge verification module
        self.knowledge_verifier = nn.Sequential(
            nn.Linear(knowledge_size, bottleneck_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(bottleneck_size, bottleneck_size)
        )

        # Project back to hidden space with corrections
        self.knowledge_to_hidden = nn.Linear(bottleneck_size, hidden_size)

        # Confidence scoring
        self.confidence_scorer = nn.Linear(bottleneck_size, 1)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, 
        hidden_states: torch.Tensor,
        knowledge_embeddings: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply knowledge-grounded verification

        Args:
            hidden_states: Model hidden states [batch, seq_len, hidden_size]
            knowledge_embeddings: Optional pre-computed knowledge embeddings
                                  [batch, seq_len, knowledge_size]
        """
        residual = hidden_states

        # Project to knowledge space
        if knowledge_embeddings is None:
            # If no knowledge provided, just project the hidden states
            knowledge_space = self.hidden_to_knowledge(hidden_states)
        else:
            # Use provided knowledge
            knowledge_space = knowledge_embeddings

        # Verify in knowledge space
        verified_knowledge = self.knowledge_verifier(knowledge_space)

        # Calculate confidence
        confidence = torch.sigmoid(self.confidence_scorer(verified_knowledge))

        # Project back to hidden space
        corrections = self.knowledge_to_hidden(verified_knowledge)

        # Apply corrections based on confidence
        corrected_states = residual + (1 - confidence) * corrections
        corrected_states = self.layer_norm(corrected_states)

        return corrected_states, confidence


class VerificationLossWithCurriculum(nn.Module):
    """
    Enhanced loss function that implements curriculum learning for verification,
    gradually increasing the difficulty of verification tasks.
    """
    def __init__(
        self, 
        task_loss_fn,
        consistency_weight: float = 0.1,
        confidence_regularization_weight: float = 0.05,
        uncertainty_weight: float = 0.02,
        curriculum_steps: int = 1000
    ):
        super().__init__()
        self.task_loss_fn = task_loss_fn
        self.consistency_weight = consistency_weight
        self.confidence_regularization_weight = confidence_regularization_weight
        self.uncertainty_weight = uncertainty_weight
        self.curriculum_steps = curriculum_steps
        self.current_step = 0

    def update_step(self):
        """Update the curriculum step counter"""
        self.current_step += 1

    def get_curriculum_factor(self):
        """
        Returns a factor between 0 and 1 that scales up as training progresses,
        implementing a curriculum that gradually increases verification difficulty
        """
        return min(self.current_step / self.curriculum_steps, 1.0)

    def forward(
        self, 
        outputs, 
        targets,
        verification_metrics: Optional[Dict] = None
    ):
        # Calculate the main task loss
        task_loss = self.task_loss_fn(outputs, targets)

        total_loss = task_loss

        # Get curriculum factor
        curriculum_factor = self.get_curriculum_factor()

        # Add verification-specific losses if metrics are provided
        if verification_metrics is not None:
            # Cross-layer consistency loss
            if "cross_layer_consistency" in verification_metrics:
                consistency = verification_metrics["cross_layer_consistency"]
                # We want to maximize consistency (minimize 1 - consistency)
                consistency_loss = torch.mean(1 - consistency)
                # Apply curriculum scaling
                weighted_consistency_loss = self.consistency_weight * curriculum_factor * consistency_loss
                total_loss += weighted_consistency_loss

            # Confidence regularization (prevent always-high or always-low confidence)
            if "layer_confidence_scores" in verification_metrics:
                confidence_scores = verification_metrics["layer_confidence_scores"]
                valid_scores = [c for c in confidence_scores if c.requires_grad]

                if valid_scores:
                    avg_confidence = torch.mean(torch.cat([conf.mean() for conf in valid_scores]))
                    # Penalize confidence scores that are too close to 0 or 1
                    confidence_reg = -torch.log(avg_confidence + 1e-10) - torch.log(1 - avg_confidence + 1e-10)
                    # Apply curriculum scaling
                    weighted_confidence_reg = self.confidence_regularization_weight * curriculum_factor * confidence_reg
                    total_loss += weighted_confidence_reg

            # Uncertainty regularization (if available)
            if "layer_uncertainty_scores" in verification_metrics:
                uncertainty_scores = verification_metrics["layer_uncertainty_scores"]
                # Early in training, encourage exploration with high uncertainty
                # Later in training, encourage certainty with low uncertainty
                target_uncertainty = max(0.5 - curriculum_factor * 0.4, 0.1)
                avg_uncertainty = torch.mean(torch.cat([unc.mean() for unc in uncertainty_scores]))
                uncertainty_reg = torch.abs(avg_uncertainty - target_uncertainty)
                weighted_uncertainty_reg = self.uncertainty_weight * curriculum_factor * uncertainty_reg
                total_loss += weighted_uncertainty_reg

        # Update curriculum step
        self.update_step()

        return total_loss


# Helper function to create a verification-enhanced model with all improvements
def create_enhanced_verification_model(
    model_name_or_path: str,
    adapter_locations: List[int] = None,
    verification_type: str = "standard",  # Options: standard, bayesian, hierarchical, residual, knowledge
    bottleneck_size: int = 64,
    enable_cross_layer: bool = True,
    freeze_base_model: bool = True,
    **kwargs
):
    """
    Creates a verification-enhanced model with various improvements

    Args:
        model_name_or_path: Hugging Face model name or path
        adapter_locations: List of layer indices where to apply verification adapters
        verification_type: Type of verification adapter to use
        bottleneck_size: Size of the verification adapter bottleneck
        enable_cross_layer: Whether to enable cross-layer verification
        freeze_base_model: Whether to freeze the base model parameters
    """
    from transformers import AutoModelForCausalLM
    from latent_verification import LatentVerificationWrapper

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    config = base_model.config

    # Determine hidden size
    hidden_size = getattr(config, 'hidden_size', 
                         getattr(config, 'd_model', 
                                getattr(config, 'n_embd', 768)))

    # Determine number of layers
    num_layers = getattr(config, 'num_hidden_layers', 
                        getattr(config, 'n_layer', 
                               getattr(config, 'num_layers', 12)))

    # Default adapter locations if not specified (every 3rd layer)
    if adapter_locations is None:
        adapter_locations = list(range(2, num_layers, 3))

    # Create wrapper with chosen verification type
    if verification_type == "bayesian":
        # Create with Bayesian verification adapters
        verification_adapters = nn.ModuleDict({
            f"layer_{layer_idx}": BayesianVerificationAdapter(
                hidden_size=hidden_size,
                bottleneck_size=bottleneck_size,
            )
            for layer_idx in adapter_locations
        })
        model = LatentVerificationWrapper(
            base_model=base_model,
            adapter_locations=adapter_locations,
            enable_cross_layer=enable_cross_layer,
            freeze_base_model=freeze_base_model,
            verification_adapters=verification_adapters
        )

    elif verification_type == "hierarchical":
        # Create with hierarchical verification adapters
        verification_adapters = nn.ModuleDict({
            f"layer_{layer_idx}": HierarchicalVerifier(
                hidden_size=hidden_size,
                bottleneck_size=bottleneck_size,
            )
            for layer_idx in adapter_locations
        })
        model = LatentVerificationWrapper(
            base_model=base_model,
            adapter_locations=adapter_locations,
            enable_cross_layer=enable_cross_layer,
            freeze_base_model=freeze_base_model,
            verification_adapters=verification_adapters
        )

    elif verification_type == "residual":
        # Create with residual verification networks
        verification_adapters = nn.ModuleDict({
            f"layer_{layer_idx}": ResidualVerificationNetwork(
                hidden_size=hidden_size,
                bottleneck_size=bottleneck_size,
            )
            for layer_idx in adapter_locations
        })
        model = LatentVerificationWrapper(
            base_model=base_model,
            adapter_locations=adapter_locations,
            enable_cross_layer=enable_cross_layer,
            freeze_base_model=freeze_base_model,
            verification_adapters=verification_adapters
        )

    elif verification_type == "knowledge":
        # Create with knowledge-grounded verification
        verification_adapters = nn.ModuleDict({
            f"layer_{layer_idx}": KnowledgeGroundedVerifier(
                hidden_size=hidden_size,
                knowledge_size=bottleneck_size * 2,
                bottleneck_size=bottleneck_size,
            )
            for layer_idx in adapter_locations
        })
        model = LatentVerificationWrapper(
            base_model=base_model,
            adapter_locations=adapter_locations,
            enable_cross_layer=enable_cross_layer,
            freeze_base_model=freeze_base_model,
            verification_adapters=verification_adapters
        )

    else:  # standard, original implementation
        from latent_verification import create_verification_model
        model = create_verification_model(
            model_name_or_path=model_name_or_path,
            adapter_locations=adapter_locations,
            bottleneck_size=bottleneck_size,
            enable_cross_layer=enable_cross_layer,
            freeze_base_model=freeze_base_model
        )

    return model

def load_bayesian_verification_model(model_path, **kwargs):
    """Load a verification model with Bayesian adapters"""
    import json
    import os
    import torch
    from transformers import AutoModelForCausalLM
    from latent_verification import LatentVerificationWrapper
    from enhanced_verification import BayesianVerificationAdapter

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    # Load verification configuration
    config_path = os.path.join(model_path, "verification_config.json")
    if not os.path.exists(config_path):
        # Try model subdirectory (common when saving with trainer)
        config_path = os.path.join(model_path, "model", "verification_config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Determine hidden size
    model_config = base_model.config
    hidden_size = getattr(model_config, 'hidden_size', 
                         getattr(model_config, 'd_model', 
                                getattr(model_config, 'n_embd', 768)))

    # Create Bayesian verification adapters
    verification_adapters = torch.nn.ModuleDict({
        f"layer_{layer_idx}": BayesianVerificationAdapter(
            hidden_size=hidden_size,
            bottleneck_size=64,
        )
        for layer_idx in config["adapter_locations"]
    })

    # Create wrapper with Bayesian adapters
    model = LatentVerificationWrapper(
        base_model=base_model,
        adapter_locations=config["adapter_locations"],
        hidden_size=config["hidden_size"],
        enable_cross_layer=config["enable_cross_layer"],
        freeze_base_model=False,
        verification_adapters=verification_adapters
    )

    # Load adapter weights
    adapter_path = os.path.join(model_path, "verification_adapters.pt")
    if not os.path.exists(adapter_path):
        adapter_path = os.path.join(model_path, "model", "verification_adapters.pt")

    model.verification_adapters.load_state_dict(
        torch.load(adapter_path)
    )

    # Load cross-layer verifier
    if config["enable_cross_layer"]:
        cl_path = os.path.join(model_path, "cross_layer_verifier.pt")
        if not os.path.exists(cl_path):
            cl_path = os.path.join(model_path, "model", "cross_layer_verifier.pt")

        model.cross_layer_verifier.load_state_dict(
            torch.load(cl_path)
        )

    return model
