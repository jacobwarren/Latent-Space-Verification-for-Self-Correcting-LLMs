#!/usr/bin/env python
"""
Embedding Space Dynamics Analyzer

This tool creates powerful visualizations of how latent verification mechanisms 
transform embeddings in high-dimensional space. It provides direct evidence for
the "thinking in latent space" hypothesis by tracking how representations evolve
through the verification process.

The analyzer creates:
1. Animated PCA/t-SNE visualizations showing representation transformations
2. Vector field plots showing the "flow" of corrections in embedding space
3. Attention pattern visualizations with verification components highlighted
4. Contrastive representation analysis comparing verified vs unverified models
5. Correction magnitude analysis across semantic content types

These visualizations provide compelling evidence for the research paper.
"""

import os, sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
from tqdm import tqdm
import argparse
from matplotlib.animation import FuncAnimation
from matplotlib import cm, colors
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import math
from scipy.spatial.distance import pdist, squareform
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch.nn.functional as F

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("embedding_dynamics.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmbeddingDynamicsAnalyzer:
    """Analyzer for visualizing embedding space transformations during verification"""

    def __init__(
        self,
        verified_model_path: str,
        base_model_name: Optional[str] = None,
        output_dir: str = "embedding_dynamics",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        high_quality: bool = False
    ):
        """
        Initialize the embedding dynamics analyzer

        Args:
            verified_model_path: Path to the verification-enhanced model
            base_model_name: Name of the base model (optional, for comparison)
            output_dir: Directory to save analysis outputs
            device: Device to run model on (cuda or cpu)
            high_quality: Whether to generate high-quality (but slower) visualizations
        """
        self.verified_model_path = verified_model_path
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.device = device
        self.high_quality = high_quality

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "animations"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "vector_fields"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "attention"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "contrastive"), exist_ok=True)

        # Will be initialized when needed
        self.tokenizer = None
        self.verified_model = None
        self.base_model = None

        # Results storage
        self.embedding_data = {}

        logger.info(f"Initializing embedding dynamics analyzer for model at {verified_model_path}")
        logger.info(f"Using device: {device}")
        logger.info(f"High quality visualizations: {high_quality}")

    def load_models(self):
        """Load the verified model and tokenizer"""
        logger.info("Loading verified model and tokenizer...")

        # Determine tokenizer source
        tokenizer_path = self.base_model_name if self.base_model_name else self.verified_model_path

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load verified model
        try:
            # First try loading with verification wrapper
            parent_dir = os.path.join(os.path.dirname(__file__), '..')
            sys.path.append(parent_dir)

            from latent_verification.enhanced_verification import load_bayesian_verification_model
            self.verified_model = load_bayesian_verification_model(self.verified_model_path).to(self.device)
            logger.info("Successfully loaded verified model with wrapper")
        except Exception as e:
            logger.warning(f"Error loading with wrapper: {e}. Trying standard loading...")
            # Fallback to standard loading
            self.verified_model = AutoModelForCausalLM.from_pretrained(
                self.verified_model_path
            ).to(self.device)
            logger.info("Loaded verified model with standard loading")

        # Load base model if specified
        if self.base_model_name:
            logger.info(f"Loading base model from {self.base_model_name}...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name
            ).to(self.device)

    def analyze_embedding_dynamics(self, input_texts: List[str] = None):
        """
        Analyze how embeddings evolve through verification

        Args:
            input_texts: List of inputs to analyze
                If None, default examples will be used
        """
        if self.verified_model is None:
            self.load_models()

        # Use default inputs if not provided
        if input_texts is None:
            input_texts = [
                "The capital of France is Berlin.",
                "The capital of France is Paris.",
                "The tallest mountain in the world is K2.",
                "The tallest mountain in the world is Mount Everest.",
                "Water boils at 50 degrees Celsius at sea level.",
                "Water boils at 100 degrees Celsius at sea level.",
                "The human heart has three chambers.",
                "The human heart has four chambers.",
                "The Earth orbits around the Sun.",
                "The Sun orbits around the Earth."
            ]

        logger.info(f"Analyzing embedding dynamics for {len(input_texts)} inputs...")

        # Reset storage
        self.embedding_data = {}

        # Process each input
        for i, text in enumerate(tqdm(input_texts, desc="Processing inputs")):
            # Get embeddings and tracks for this input
            embedding_tracks = self._extract_embedding_dynamics(text)

            # Store data
            self.embedding_data[text] = embedding_tracks

        # Create visualizations
        self._create_embedding_animations()
        self._create_vector_field_plots()
        self._visualize_embedding_distances()

        # Save embedding data
        data_to_save = {}
        for text, tracks in self.embedding_data.items():
            # Convert torch tensors to lists for JSON serialization
            processed_tracks = {
                "verification_confidences": [float(conf) for conf in tracks["verification_confidences"]],
                "adapter_locations": tracks["adapter_locations"],
                "correction_magnitudes": [float(mag) for mag in tracks["correction_magnitudes"]]
            }
            data_to_save[text] = processed_tracks

        with open(os.path.join(self.output_dir, "embedding_dynamics.json"), "w") as f:
            json.dump(data_to_save, f, indent=2)

        return self.embedding_data

    def _extract_embedding_dynamics(self, text: str) -> Dict[str, Any]:
        """
        Extract embedding dynamics through layers for a given input

        Args:
            text: The input text

        Returns:
            Dictionary containing embedding trajectories and metadata
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Initialize variables to store the original states
        old_all_adapter_hidden_states = None
        old_all_confidence_scores = None

        # Save original states if they exist
        if hasattr(self.verified_model, 'all_adapter_hidden_states'):
            old_all_adapter_hidden_states = self.verified_model.all_adapter_hidden_states
            self.verified_model.all_adapter_hidden_states = []

        if hasattr(self.verified_model, 'all_confidence_scores'):
            old_all_confidence_scores = self.verified_model.all_confidence_scores
            self.verified_model.all_confidence_scores = []

        # Forward pass with verified model
        with torch.no_grad():
            outputs = self.verified_model(**inputs, output_hidden_states=True, output_attentions=True)

        # Get hidden states
        hidden_states = [h.detach().cpu() for h in outputs.hidden_states]

        # Get adapter locations
        adapter_locations = []
        if hasattr(self.verified_model, 'adapter_locations'):
            adapter_locations = self.verified_model.adapter_locations

        # Get confidence scores
        verification_confidences = []
        if hasattr(self.verified_model, 'all_confidence_scores') and self.verified_model.all_confidence_scores:
            for conf in self.verified_model.all_confidence_scores:
                verification_confidences.append(conf.mean().item())

        # Calculate correction magnitudes
        correction_magnitudes = []
        if hasattr(self.verified_model, 'all_adapter_hidden_states') and len(self.verified_model.all_adapter_hidden_states) > 0:
            adapter_hidden_states = self.verified_model.all_adapter_hidden_states

            for i, (adapter_idx, corrected_state) in enumerate(zip(adapter_locations, adapter_hidden_states)):
                if adapter_idx < len(hidden_states):
                    # Get the original hidden state before correction
                    orig_state = hidden_states[adapter_idx]

                    # Calculate L2 norm of the difference
                    diff = (corrected_state.cpu() - orig_state).norm().item()
                    # Normalize by tensor size
                    normalized_diff = diff / (orig_state.numel() ** 0.5)
                    correction_magnitudes.append(normalized_diff)

        # Get attention patterns
        attention_patterns = None
        if outputs.attentions:
            attention_patterns = [attn.detach().cpu() for attn in outputs.attentions]

        # Collect base model data if available
        base_hidden_states = None
        if self.base_model:
            with torch.no_grad():
                base_outputs = self.base_model(**inputs, output_hidden_states=True)
                base_hidden_states = [h.detach().cpu() for h in base_outputs.hidden_states]

        # Restore adapter states if they were saved
        if hasattr(self.verified_model, 'all_adapter_hidden_states') and old_all_adapter_hidden_states is not None:
            self.verified_model.all_adapter_hidden_states = old_all_adapter_hidden_states

        if hasattr(self.verified_model, 'all_confidence_scores') and old_all_confidence_scores is not None:
            self.verified_model.all_confidence_scores = old_all_confidence_scores

        # Return collected data
        return {
            "hidden_states": hidden_states,
            "base_hidden_states": base_hidden_states,
            "adapter_locations": adapter_locations,
            "verification_confidences": verification_confidences,
            "correction_magnitudes": correction_magnitudes,
            "attention_patterns": attention_patterns
        }

    def _create_embedding_animations(self):
        """Create animations showing how embeddings evolve through layers"""
        logger.info("Creating embedding animations...")

        # Group inputs into pairs of contradicting statements
        paired_inputs = []
        all_texts = list(self.embedding_data.keys())

        for i in range(0, len(all_texts), 2):
            if i + 1 < len(all_texts):
                paired_inputs.append((all_texts[i], all_texts[i+1]))

        # If odd number of inputs, add the last one individually
        if len(all_texts) % 2 != 0:
            paired_inputs.append((all_texts[-1],))

        # Process each pair
        for pair_idx, text_pair in enumerate(paired_inputs):
            # Create animation for this pair
            self._create_pair_animation(pair_idx, text_pair)

            # Create individual PCA plots for key layers
            self._create_layer_pca_plots(pair_idx, text_pair)

        # Create aggregate embeddings animation with all inputs
        self._create_aggregate_animation(all_texts)

    def _create_pair_animation(self, pair_idx: int, text_pair: Tuple[str, ...]):
        """
        Create animation for a pair of contradicting statements

        Args:
            pair_idx: Index of the pair
            text_pair: Tuple of text inputs (usually 2, but can be 1)
        """
        # Get hidden states for each text
        pair_hidden_states = [self.embedding_data[text]["hidden_states"] for text in text_pair]

        # Get adapter locations (should be the same for all texts)
        adapter_locations = self.embedding_data[text_pair[0]]["adapter_locations"]

        # Get maximum number of layers
        max_layers = max(len(states) for states in pair_hidden_states)

        # Prepare data for dimensionality reduction
        combined_states = []

        # Collect [CLS] or first token representation from each layer of each text
        for layer_idx in range(max_layers):
            for text_idx, states in enumerate(pair_hidden_states):
                if layer_idx < len(states):
                    # Use the mean over all tokens for better representation
                    combined_states.append(states[layer_idx].mean(dim=1).numpy())

        # Apply PCA
        pca = PCA(n_components=2)
        reduced_states = pca.fit_transform(np.vstack(combined_states))

        # Split back into per-text, per-layer
        text_layer_states = []

        idx = 0
        for text_idx in range(len(text_pair)):
            text_states = []
            for layer_idx in range(max_layers):
                if layer_idx < len(pair_hidden_states[text_idx]):
                    text_states.append(reduced_states[idx])
                    idx += 1
            text_layer_states.append(text_states)

        # Create animation
        fig, ax = plt.subplots(figsize=(10, 8))

        def update(frame):
            ax.clear()

            # Plot the trajectories up to the current frame
            for text_idx, text_states in enumerate(text_layer_states):
                # Use different colors for different texts
                color = 'red' if text_idx % 2 == 0 else 'green'
                label = f"Text {text_idx + 1}: {text_pair[text_idx][:30]}..." if len(text_pair[text_idx]) > 30 else text_pair[text_idx]

                # Plot trajectory up to current frame
                if frame > 0:
                    ax.plot(
                        [state[0] for state in text_states[:frame]],
                        [state[1] for state in text_states[:frame]],
                        color=color, alpha=0.5
                    )

                # Plot point at current frame if within range
                if frame < len(text_states):
                    ax.scatter(
                        text_states[frame][0], text_states[frame][1],
                        color=color, s=100, label=label
                    )

            # Mark adapter layers with a different style
            if frame in adapter_locations:
                ax.set_facecolor("#f0f0f0")
                # Add text indicating verification layer
                ax.text(
                    0.5, 0.02, f"Verification Layer {frame}",
                    transform=ax.transAxes, fontsize=12, ha='center',
                    bbox=dict(facecolor='yellow', alpha=0.5)
                )
            else:
                ax.set_facecolor("white")

            ax.set_title(f"Embedding Evolution Through Layers (Layer {frame})")
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
            ax.legend(loc="upper right")
            ax.grid(alpha=0.3)

            return ax

        # Create animation
        anim = FuncAnimation(
            fig, update, frames=max_layers,
            interval=200, blit=False
        )

        # Save animation
        output_path = os.path.join(self.output_dir, "animations", f"pair_{pair_idx+1}_pca_evolution.gif")
        anim.save(output_path, writer='pillow', fps=5, dpi=100)
        plt.close()

        logger.info(f"Saved PCA evolution animation to {output_path}")

        # If high quality flag is set, also create a t-SNE animation (slower but sometimes better)
        if self.high_quality:
            self._create_tsne_animation(pair_idx, text_pair, pair_hidden_states, adapter_locations)

    def _create_tsne_animation(self, pair_idx: int, text_pair: Tuple[str, ...], pair_hidden_states: List[List[torch.Tensor]], adapter_locations: List[int]):
        """
        Create t-SNE animation (higher quality but slower)

        Args:
            pair_idx: Index of the pair
            text_pair: Tuple of text inputs
            pair_hidden_states: Hidden states for each text
            adapter_locations: Locations of adapter layers
        """
        logger.info(f"Creating t-SNE animation for pair {pair_idx+1}...")

        # Get maximum number of layers
        max_layers = max(len(states) for states in pair_hidden_states)

        # Create figure
        fig, axes = plt.subplots(
            math.ceil(max_layers / 6), 6,
            figsize=(18, 3 * math.ceil(max_layers / 6)),
            constrained_layout=True
        )
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        # For each layer, create a t-SNE plot
        for layer_idx in range(max_layers):
            # Collect states from this layer for all texts
            layer_states = []
            text_indices = []

            for text_idx, states in enumerate(pair_hidden_states):
                if layer_idx < len(states):
                    # Get all token representations for this layer
                    tokens = states[layer_idx].squeeze(0)
                    layer_states.append(tokens)
                    text_indices.extend([text_idx] * tokens.shape[0])

            # Combine all tokens from all texts for this layer
            combined = torch.cat(layer_states, dim=0).numpy()

            # Skip if too few points
            if combined.shape[0] < 4:
                continue

            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, combined.shape[0] // 2))
            reduced = tsne.fit_transform(combined)

            # Plot in the appropriate subplot
            ax = axes[layer_idx]

            # Plot points
            for text_idx in set(text_indices):
                text_mask = np.array(text_indices) == text_idx
                color = 'red' if text_idx % 2 == 0 else 'green'

                ax.scatter(
                    reduced[text_mask, 0], reduced[text_mask, 1],
                    color=color, alpha=0.7, s=30
                )

            # Mark adapter layers with a different style
            if layer_idx in adapter_locations:
                ax.set_facecolor("#f0f0f0")
                # Add text indicating verification layer
                ax.text(
                    0.5, 0.02, "Verification",
                    transform=ax.transAxes, fontsize=9, ha='center',
                    bbox=dict(facecolor='yellow', alpha=0.5)
                )

            ax.set_title(f"Layer {layer_idx}")
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused subplots
        for i in range(max_layers, len(axes)):
            axes[i].axis('off')

        # Add legend
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10)
        ]
        fig.legend(
            handles,
            [text[:30] + "..." if len(text) > 30 else text for text in text_pair],
            loc='upper center', bbox_to_anchor=(0.5, 0.98),
            ncol=2, fontsize=12
        )

        plt.suptitle("t-SNE Visualization of Token Embeddings Across Layers", fontsize=16, y=0.99)

        # Save figure
        output_path = os.path.join(self.output_dir, "animations", f"pair_{pair_idx+1}_tsne_layers.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved t-SNE layer visualization to {output_path}")

    def _create_layer_pca_plots(self, pair_idx: int, text_pair: Tuple[str, ...]):
        """
        Create PCA plots for key layers

        Args:
            pair_idx: Index of the pair
            text_pair: Tuple of text inputs
        """
        # Get hidden states for each text
        pair_hidden_states = [self.embedding_data[text]["hidden_states"] for text in text_pair]

        # Get adapter locations
        adapter_locations = self.embedding_data[text_pair[0]]["adapter_locations"]

        # Select key layers to visualize
        key_layers = []

        # Include first, last, and all adapter layers
        key_layers.append(0)  # First layer
        key_layers.extend(adapter_locations)  # Adapter layers
        key_layers.append(max(len(states) - 1 for states in pair_hidden_states))  # Last layer

        # Remove duplicates and sort
        key_layers = sorted(set(key_layers))

        # Create figure with subplots for each key layer
        num_layers = len(key_layers)
        fig, axes = plt.subplots(
            1, num_layers,
            figsize=(4 * num_layers, 4),
            constrained_layout=True
        )

        # Handle case of single subplot
        if num_layers == 1:
            axes = [axes]

        # Process each key layer
        for i, layer_idx in enumerate(key_layers):
            # Collect states from this layer for all texts
            layer_states = []
            text_labels = []

            for text_idx, states in enumerate(pair_hidden_states):
                if layer_idx < len(states):
                    # Get mean token representation for this layer
                    mean_state = states[layer_idx].mean(dim=1).numpy()
                    layer_states.append(mean_state)
                    text_labels.append(text_idx)

            # Combine all representations
            combined = np.vstack(layer_states)

            # Apply PCA if enough samples
            if combined.shape[0] > 1:
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(combined)

                # Plot in the appropriate subplot
                ax = axes[i]

                # Plot points
                for j, (text_idx, point) in enumerate(zip(text_labels, reduced)):
                    color = 'red' if text_idx % 2 == 0 else 'green'
                    label = f"Text {text_idx + 1}" if j == 0 or j == len(text_labels) // 2 else None

                    ax.scatter(
                        point[0], point[1],
                        color=color, s=100, label=label
                    )

                    # Add text annotation
                    short_text = text_pair[text_idx][:15] + "..." if len(text_pair[text_idx]) > 15 else text_pair[text_idx]
                    ax.annotate(
                        short_text,
                        (point[0], point[1]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center'
                    )

                # Mark adapter layers with a different style
                if layer_idx in adapter_locations:
                    ax.set_facecolor("#f0f0f0")
                    ax.set_title(f"Layer {layer_idx}\n(Verification)")
                else:
                    ax.set_title(f"Layer {layer_idx}")

                ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
                ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")

                # Only add legend to first subplot
                if i == 0:
                    ax.legend()
            else:
                # Not enough samples
                axes[i].text(0.5, 0.5, f"Layer {layer_idx}\nInsufficient data",
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=axes[i].transAxes)
                axes[i].axis('off')

        plt.suptitle("PCA of Representations at Key Layers", fontsize=16)

        # Save figure
        output_path = os.path.join(self.output_dir, "animations", f"pair_{pair_idx+1}_key_layers.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved key layer PCA plots to {output_path}")

    def _create_aggregate_animation(self, all_texts: List[str]):
        """
        Create aggregate animation with all inputs

        Args:
            all_texts: List of all text inputs
        """
        logger.info("Creating aggregate embedding animation...")

        # Determine if texts can be categorized as true/false
        truth_indicators = ["Paris", "Mount Everest", "100 degrees", "four chambers", "Earth orbits", "is not visible"]
        falsehood_indicators = ["Berlin", "K2", "50 degrees", "three chambers", "Sun orbits", "is visible"]

        # Categorize texts
        text_categories = []
        for text in all_texts:
            if any(indicator in text for indicator in truth_indicators):
                text_categories.append("true")
            elif any(indicator in text for indicator in falsehood_indicators):
                text_categories.append("false")
            else:
                text_categories.append("neutral")

        # Get hidden states for each text
        all_hidden_states = [self.embedding_data[text]["hidden_states"] for text in all_texts]

        # Get adapter locations (should be the same for all texts)
        adapter_locations = self.embedding_data[all_texts[0]]["adapter_locations"]

        # Get maximum number of layers
        max_layers = max(len(states) for states in all_hidden_states)

        # Prepare data for dimensionality reduction
        combined_states = []
        state_metadata = []  # Store metadata about each state (text_idx, layer_idx)

        # Collect mean representation from each layer of each text
        for text_idx, states in enumerate(all_hidden_states):
            for layer_idx in range(len(states)):
                # Use the mean over all tokens
                combined_states.append(states[layer_idx].mean(dim=1).numpy())
                state_metadata.append({
                    "text_idx": text_idx,
                    "layer_idx": layer_idx,
                    "category": text_categories[text_idx]
                })

        # Apply PCA
        pca = PCA(n_components=2)
        reduced_states = pca.fit_transform(np.vstack(combined_states))

        # Create figure
        fig = plt.figure(figsize=(12, 10))

        # Create scatter plot
        ax = fig.add_subplot(111)

        # Create animation
        def update(frame):
            ax.clear()

            # Plot states from all texts at the current layer
            current_layer_indices = [i for i, meta in enumerate(state_metadata) if meta["layer_idx"] == frame]

            if not current_layer_indices:
                return ax

            # Plot points
            for i in current_layer_indices:
                meta = state_metadata[i]
                text_idx = meta["text_idx"]
                category = meta["category"]

                # Choose color based on category
                if category == "true":
                    color = 'green'
                elif category == "false":
                    color = 'red'
                else:
                    color = 'blue'

                # Add to plot
                ax.scatter(
                    reduced_states[i, 0], reduced_states[i, 1],
                    color=color, s=100, alpha=0.7,
                    label=category if text_idx % len(text_categories) == 0 else None
                )

                # Add text label
                short_text = all_texts[text_idx][:20] + "..." if len(all_texts[text_idx]) > 20 else all_texts[text_idx]
                ax.annotate(
                    short_text,
                    (reduced_states[i, 0], reduced_states[i, 1]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=8
                )

            # Mark adapter layers with a different style
            if frame in adapter_locations:
                ax.set_facecolor("#f0f0f0")
                # Add text indicating verification layer
                ax.text(
                    0.5, 0.02, f"Verification Layer {frame}",
                    transform=ax.transAxes, fontsize=12, ha='center',
                    bbox=dict(facecolor='yellow', alpha=0.5)
                )
            else:
                ax.set_facecolor("white")

            # Build title with additional information
            if frame in adapter_locations:
                confidence_info = ""
                for text_idx, text in enumerate(all_texts):
                    if "verification_confidences" in self.embedding_data[text]:
                        confidences = self.embedding_data[text]["verification_confidences"]
                        adapter_idx = adapter_locations.index(frame)
                        if adapter_idx < len(confidences):
                            conf = confidences[adapter_idx]
                            short_text = text[:10] + "..." if len(text) > 10 else text
                            confidence_info += f"{short_text}: {conf:.3f} "

                if confidence_info:
                    confidence_title = f"\nConfidence Scores: {confidence_info}"
                else:
                    confidence_title = ""

                ax.set_title(f"Layer {frame} (Verification){confidence_title}")
            else:
                ax.set_title(f"Layer {frame}")

            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")

            # Add legend with unique categories
            handles = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='True Statements'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='False Statements')
            ]
            if 'neutral' in text_categories:
                handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Neutral'))

            ax.legend(handles=handles, loc='upper right')
            ax.grid(alpha=0.3)

            return ax

        # Create animation
        anim = FuncAnimation(
            fig, update, frames=max_layers,
            interval=200, blit=False
        )

        # Save animation
        output_path = os.path.join(self.output_dir, "animations", "aggregate_embedding_evolution.gif")
        anim.save(output_path, writer='pillow', fps=5, dpi=100)
        plt.close()

        logger.info(f"Saved aggregate embedding animation to {output_path}")

        # Create interactive version with Plotly if high_quality is set
        if self.high_quality:
            self._create_interactive_embedding_animation(all_texts, all_hidden_states, adapter_locations, text_categories)

    def _create_interactive_embedding_animation(self, all_texts: List[str], all_hidden_states: List[List[torch.Tensor]], adapter_locations: List[int], text_categories: List[str]):
        """
        Create interactive embedding animation with Plotly

        Args:
            all_texts: List of all text inputs
            all_hidden_states: Hidden states for each text
            adapter_locations: Locations of adapter layers
            text_categories: Category (true/false/neutral) for each text
        """
        logger.info("Creating interactive embedding visualization...")

        # Get maximum number of layers
        max_layers = max(len(states) for states in all_hidden_states)

        # Prepare data for dimensionality reduction
        combined_states = []
        state_metadata = []  # Store metadata about each state (text_idx, layer_idx)

        # Collect mean representation from each layer of each text
        for text_idx, states in enumerate(all_hidden_states):
            for layer_idx in range(len(states)):
                # Use the mean over all tokens
                combined_states.append(states[layer_idx].mean(dim=1).numpy())
                state_metadata.append({
                    "text_idx": text_idx,
                    "layer_idx": layer_idx,
                    "category": text_categories[text_idx],
                    "text": all_texts[text_idx],
                    "is_adapter": layer_idx in adapter_locations
                })

        # Apply PCA
        pca = PCA(n_components=3)  # Use 3D for interactive visualization
        reduced_states = pca.fit_transform(np.vstack(combined_states))

        # Create DataFrame for Plotly
        df = pd.DataFrame({
            "PC1": reduced_states[:, 0],
            "PC2": reduced_states[:, 1],
            "PC3": reduced_states[:, 2] if reduced_states.shape[1] > 2 else np.zeros(reduced_states.shape[0]),
            "Text": [meta["text"] for meta in state_metadata],
            "Layer": [meta["layer_idx"] for meta in state_metadata],
            "Category": [meta["category"] for meta in state_metadata],
            "IsAdapter": [meta["is_adapter"] for meta in state_metadata],
            "TextIndex": [meta["text_idx"] for meta in state_metadata]
        })

        # Create interactive 3D scatter plot
        fig = px.scatter_3d(
            df, x="PC1", y="PC2", z="PC3",
            color="Category", symbol="IsAdapter",
            hover_name="Text", hover_data=["Layer", "TextIndex"],
            animation_frame="Layer", 
            color_discrete_map={"true": "green", "false": "red", "neutral": "blue"},
            symbol_map={True: "diamond", False: "circle"},
            title="Interactive Embedding Space Evolution",
            labels={
                "PC1": f"PC1 ({pca.explained_variance_ratio_[0]:.2%})",
                "PC2": f"PC2 ({pca.explained_variance_ratio_[1]:.2%})",
                "PC3": f"PC3 ({pca.explained_variance_ratio_[2]:.2%})" if reduced_states.shape[1] > 2 else "PC3"
            }
        )

        # Update layout for better visibility
        fig.update_layout(
            scene=dict(
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.2%})",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.2%})",
                zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.2%})" if reduced_states.shape[1] > 2 else "PC3",
            ),
            legend_title_text="Category & Layer Type",
            margin=dict(r=20, l=10, b=10, t=60)
        )

        # Save as HTML
        output_path = os.path.join(self.output_dir, "animations", "interactive_embedding_evolution.html")
        fig.write_html(output_path)

        logger.info(f"Saved interactive embedding visualization to {output_path}")

    def _create_vector_field_plots(self):
        """Create vector field plots showing the flow of corrections in embedding space"""
        logger.info("Creating vector field plots...")

        # Check if we have any adapter hidden states
        has_adapter_states = False
        for text, data in self.embedding_data.items():
            if "adapter_locations" in data and data["adapter_locations"]:
                has_adapter_states = True
                break

        if not has_adapter_states:
            logger.warning("No adapter hidden states found. Skipping vector field plots.")
            return

        # Group inputs into pairs of contradicting statements
        paired_inputs = []
        all_texts = list(self.embedding_data.keys())

        for i in range(0, len(all_texts), 2):
            if i + 1 < len(all_texts):
                paired_inputs.append((all_texts[i], all_texts[i+1]))

        # Process each pair
        for pair_idx, text_pair in enumerate(paired_inputs):
            # Create vector field plot for this pair
            self._create_correction_vector_field(pair_idx, text_pair)

        # Create aggregate vector field with all inputs
        self._create_aggregate_vector_field(all_texts)

    def _create_correction_vector_field(self, pair_idx: int, text_pair: Tuple[str, str]):
        """
        Create vector field plot showing corrections for a pair of texts

        Args:
            pair_idx: Index of the pair
            text_pair: Tuple of text inputs
        """
        # Get hidden states and adapter locations
        pair_hidden_states = [self.embedding_data[text]["hidden_states"] for text in text_pair]
        adapter_locations = self.embedding_data[text_pair[0]]["adapter_locations"]

        # Skip if no adapter locations
        if not adapter_locations:
            return

        # Create figure with subplots for each adapter layer
        fig, axes = plt.subplots(
            1, len(adapter_locations),
            figsize=(5 * len(adapter_locations), 5),
            constrained_layout=True
        )

        # Handle case of single subplot
        if len(adapter_locations) == 1:
            axes = [axes]

        # Process each adapter layer
        for i, adapter_idx in enumerate(adapter_locations):
            ax = axes[i]

            # Collect pre and post states for this adapter
            pre_states = []
            post_states = []
            text_indices = []

            for text_idx, text in enumerate(text_pair):
                data = self.embedding_data[text]

                # Skip if adapter index is out of range
                if adapter_idx >= len(data["hidden_states"]):
                    continue

                # Get original (pre-correction) state
                pre_state = data["hidden_states"][adapter_idx].mean(dim=1).numpy()

                # Check if we have adapter hidden states
                if "all_adapter_hidden_states" in data:
                    adapter_states = data["all_adapter_hidden_states"]
                    adapter_loc_idx = data["adapter_locations"].index(adapter_idx)

                    if adapter_loc_idx < len(adapter_states):
                        # Get corrected (post-correction) state
                        post_state = adapter_states[adapter_loc_idx].mean(dim=1).cpu().numpy()

                        pre_states.append(pre_state)
                        post_states.append(post_state)
                        text_indices.append(text_idx)

            # Skip if no data for this adapter
            if not pre_states:
                ax.text(0.5, 0.5, f"Adapter {adapter_idx}\nNo data",
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes)
                ax.axis('off')
                continue

            # Combine states for dimensionality reduction
            combined_states = np.vstack(pre_states + post_states)

            # Apply PCA
            pca = PCA(n_components=2)
            reduced_states = pca.fit_transform(combined_states)

            # Split back into pre and post states
            reduced_pre = reduced_states[:len(pre_states)]
            reduced_post = reduced_states[len(pre_states):]

            # Plot pre and post states
            for j, (pre, post, text_idx) in enumerate(zip(reduced_pre, reduced_post, text_indices)):
                # Choose color based on text index
                color = 'red' if text_idx % 2 == 0 else 'green'

                # Plot the pre state
                ax.scatter(pre[0], pre[1], color=color, s=100, alpha=0.5, label=f"Pre {text_idx+1}" if j == 0 or j == len(text_indices) // 2 else None)

                # Plot the post state
                ax.scatter(post[0], post[1], color=color, s=100, marker='x', label=f"Post {text_idx+1}" if j == 0 or j == len(text_indices) // 2 else None)

                # Draw arrow from pre to post
                ax.arrow(pre[0], pre[1], post[0] - pre[0], post[1] - pre[1],
                       color=color, width=0.01, head_width=0.03, head_length=0.03, alpha=0.7)

                # Add text label (shortened)
                short_text = text_pair[text_idx][:15] + "..." if len(text_pair[text_idx]) > 15 else text_pair[text_idx]
                ax.annotate(
                    short_text,
                    (pre[0], pre[1]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=8
                )

            # Set title and labels
            ax.set_title(f"Adapter Layer {adapter_idx} Corrections")
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
            ax.legend()
            ax.grid(alpha=0.3)

        plt.suptitle("Correction Vector Fields", fontsize=16)

        # Save figure
        output_path = os.path.join(self.output_dir, "vector_fields", f"pair_{pair_idx+1}_corrections.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved correction vector field to {output_path}")

    def _create_aggregate_vector_field(self, all_texts: List[str]):
        """
        Create aggregate vector field with all inputs

        Args:
            all_texts: List of all text inputs
        """
        # Determine if texts can be categorized as true/false
        truth_indicators = ["Paris", "Mount Everest", "100 degrees", "four chambers", "Earth orbits", "is not visible"]
        falsehood_indicators = ["Berlin", "K2", "50 degrees", "three chambers", "Sun orbits", "is visible"]

        # Categorize texts
        text_categories = []
        for text in all_texts:
            if any(indicator in text for indicator in truth_indicators):
                text_categories.append("true")
            elif any(indicator in text for indicator in falsehood_indicators):
                text_categories.append("false")
            else:
                text_categories.append("neutral")

        # Get adapter locations (should be the same for all texts)
        adapter_locations = self.embedding_data[all_texts[0]]["adapter_locations"]

        # Skip if no adapter locations
        if not adapter_locations:
            return

        # Process key adapter layers (choose middle adapter layer by default)
        middle_adapter_idx = adapter_locations[len(adapter_locations) // 2]

        # Collect pre and post states for this adapter
        pre_states = []
        post_states = []
        categories = []
        text_shorts = []

        for text_idx, text in enumerate(all_texts):
            data = self.embedding_data[text]

            # Skip if adapter index is out of range
            if middle_adapter_idx >= len(data["hidden_states"]):
                continue

            # Get original (pre-correction) state
            pre_state = data["hidden_states"][middle_adapter_idx].mean(dim=1).numpy()

            # Check if we have adapter hidden states
            if hasattr(self.verified_model, 'all_adapter_hidden_states'):
                # Find the index of the middle adapter in adapter_locations
                adapter_loc_idx = data["adapter_locations"].index(middle_adapter_idx)

                # Get all_adapter_hidden_states from model
                model_adapter_states = self.verified_model.all_adapter_hidden_states

                if adapter_loc_idx < len(model_adapter_states):
                    # Get corrected (post-correction) state
                    post_state = model_adapter_states[adapter_loc_idx].mean(dim=1).cpu().numpy()

                    pre_states.append(pre_state)
                    post_states.append(post_state)
                    categories.append(text_categories[text_idx])
                    text_shorts.append(text[:15] + "..." if len(text) > 15 else text)

        # Skip if no data
        if not pre_states:
            logger.warning("No adapter states found for vector field. Skipping aggregate vector field.")
            return

        # Combine states for dimensionality reduction
        combined_states = np.vstack(pre_states + post_states)

        # Apply PCA
        pca = PCA(n_components=2)
        reduced_states = pca.fit_transform(combined_states)

        # Split back into pre and post states
        reduced_pre = reduced_states[:len(pre_states)]
        reduced_post = reduced_states[len(pre_states):]

        # Create figure
        plt.figure(figsize=(12, 10))

        # Plot vector field
        for i, (pre, post, category, text) in enumerate(zip(reduced_pre, reduced_post, categories, text_shorts)):
            # Choose color based on category
            if category == "true":
                color = 'green'
            elif category == "false":
                color = 'red'
            else:
                color = 'blue'

            # Plot the pre state
            plt.scatter(pre[0], pre[1], color=color, s=100, alpha=0.5)

            # Plot the post state
            plt.scatter(post[0], post[1], color=color, s=100, marker='x')

            # Draw arrow from pre to post
            plt.arrow(pre[0], pre[1], post[0] - pre[0], post[1] - pre[1],
                   color=color, width=0.01, head_width=0.05, head_length=0.05, alpha=0.7)

            # Add text label
            plt.annotate(
                text,
                (pre[0], pre[1]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=10
            )

        # Add legend
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='True Statements'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='False Statements')
        ]
        if 'neutral' in categories:
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Neutral'))

        plt.legend(handles=handles, loc='upper right')

        # Set title and labels
        plt.title(f"Verification Corrections at Layer {middle_adapter_idx}")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        plt.grid(alpha=0.3)

        # Save figure
        output_path = os.path.join(self.output_dir, "vector_fields", "aggregate_corrections.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved aggregate correction vector field to {output_path}")

        # If high_quality is set, create an interactive version
        if self.high_quality:
            self._create_interactive_vector_field(all_texts, middle_adapter_idx, text_categories)

    def _create_interactive_vector_field(self, all_texts: List[str], adapter_idx: int, text_categories: List[str]):
        """
        Create interactive vector field visualization with Plotly

        Args:
            all_texts: List of all text inputs
            adapter_idx: Index of the adapter layer to visualize
            text_categories: Category (true/false/neutral) for each text
        """
        logger.info("Creating interactive vector field visualization...")

        # Collect pre and post states for this adapter
        pre_states = []
        post_states = []
        categories = []
        confidence_scores = []
        text_shorts = []

        for text_idx, text in enumerate(all_texts):
            data = self.embedding_data[text]

            # Skip if adapter index is out of range
            if adapter_idx >= len(data["hidden_states"]):
                continue

            # Get original (pre-correction) state
            pre_state = data["hidden_states"][adapter_idx].mean(dim=1).numpy()

            # Check if we have adapter hidden states
            if hasattr(self.verified_model, 'all_adapter_hidden_states'):
                # Find the index of the adapter in adapter_locations
                adapter_loc_idx = data["adapter_locations"].index(adapter_idx)

                # Get all_adapter_hidden_states from model
                model_adapter_states = self.verified_model.all_adapter_hidden_states

                if adapter_loc_idx < len(model_adapter_states):
                    # Get corrected (post-correction) state
                    post_state = model_adapter_states[adapter_loc_idx].mean(dim=1).cpu().numpy()

                    pre_states.append(pre_state)
                    post_states.append(post_state)
                    categories.append(text_categories[text_idx])
                    text_shorts.append(text)

                    # Get confidence score if available
                    conf_score = None
                    if "verification_confidences" in data and adapter_loc_idx < len(data["verification_confidences"]):
                        conf_score = data["verification_confidences"][adapter_loc_idx]
                    confidence_scores.append(conf_score)

        # Skip if no data
        if not pre_states:
            return

        # Combine states for dimensionality reduction
        combined_states = np.vstack(pre_states + post_states)

        # Apply PCA
        pca = PCA(n_components=3)  # Use 3D for interactive visualization
        reduced_states = pca.fit_transform(combined_states)

        # Split back into pre and post states
        reduced_pre = reduced_states[:len(pre_states)]
        reduced_post = reduced_states[len(pre_states):]

        # Calculate vectors (directions and magnitudes)
        vectors = reduced_post - reduced_pre
        magnitudes = np.sqrt(np.sum(vectors**2, axis=1))

        # Create DataFrame for Plotly
        df = pd.DataFrame({
            "PreX": reduced_pre[:, 0],
            "PreY": reduced_pre[:, 1],
            "PreZ": reduced_pre[:, 2] if reduced_pre.shape[1] > 2 else np.zeros(len(reduced_pre)),
            "PostX": reduced_post[:, 0],
            "PostY": reduced_post[:, 1],
            "PostZ": reduced_post[:, 2] if reduced_post.shape[1] > 2 else np.zeros(len(reduced_post)),
            "VectorX": vectors[:, 0],
            "VectorY": vectors[:, 1],
            "VectorZ": vectors[:, 2] if vectors.shape[1] > 2 else np.zeros(len(vectors)),
            "Magnitude": magnitudes,
            "Text": text_shorts,
            "Category": categories,
            "Confidence": confidence_scores
        })

        # Create interactive 3D scatter plot
        fig = go.Figure()

        # Add a trace for each category
        for category in set(categories):
            category_df = df[df["Category"] == category]

            # Choose color based on category
            if category == "true":
                color = 'green'
            elif category == "false":
                color = 'red'
            else:
                color = 'blue'

            # Add pre-correction points
            fig.add_trace(go.Scatter3d(
                x=category_df["PreX"],
                y=category_df["PreY"],
                z=category_df["PreZ"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=color,
                    opacity=0.7
                ),
                text=category_df["Text"],
                name=f"{category.title()} Pre-Correction"
            ))

            # Add post-correction points
            fig.add_trace(go.Scatter3d(
                x=category_df["PostX"],
                y=category_df["PostY"],
                z=category_df["PostZ"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=color,
                    symbol="x",
                    opacity=0.7
                ),
                text=category_df["Text"],
                name=f"{category.title()} Post-Correction"
            ))

            # Add arrows
            for _, row in category_df.iterrows():
                # Add arrow (implemented as cone)
                fig.add_trace(go.Cone(
                    x=[row["PreX"]],
                    y=[row["PreY"]],
                    z=[row["PreZ"]],
                    u=[row["VectorX"]],
                    v=[row["VectorY"]],
                    w=[row["VectorZ"]],
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    sizemode="absolute",
                    sizeref=0.5
                ))

                # Add a trace for the confidence score if available
                if row["Confidence"] is not None:
                    fig.add_trace(go.Scatter3d(
                        x=[row["PreX"]],
                        y=[row["PreY"]],
                        z=[row["PreZ"]],
                        mode="text",
                        text=[f"Conf: {row['Confidence']:.3f}"],
                        textposition="top center",
                        textfont=dict(
                            color=color,
                            size=10
                        ),
                        showlegend=False
                    ))

        # Update layout
        fig.update_layout(
            title=f"Interactive Vector Field at Layer {adapter_idx}",
            scene=dict(
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.2%})",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.2%})",
                zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.2%})" if reduced_pre.shape[1] > 2 else "PC3",
            ),
            margin=dict(r=20, l=10, b=10, t=60)
        )

        # Save as HTML
        output_path = os.path.join(self.output_dir, "vector_fields", "interactive_vector_field.html")
        fig.write_html(output_path)

        logger.info(f"Saved interactive vector field to {output_path}")

    def _visualize_embedding_distances(self):
        """Visualize distances between embeddings of true and false statements"""
        logger.info("Visualizing embedding distances...")

        # Group inputs into pairs of contradicting statements
        paired_inputs = []
        all_texts = list(self.embedding_data.keys())

        for i in range(0, len(all_texts), 2):
            if i + 1 < len(all_texts):
                paired_inputs.append((all_texts[i], all_texts[i+1]))

        # Skip if no pairs
        if not paired_inputs:
            return

        # Get maximum number of layers
        max_layers = 0
        for text in all_texts:
            max_layers = max(max_layers, len(self.embedding_data[text]["hidden_states"]))

        # Initialize distance matrices
        base_distances = np.zeros((len(paired_inputs), max_layers))
        verified_distances = np.zeros((len(paired_inputs), max_layers))

        # Calculate distances for each pair and layer
        for pair_idx, (text1, text2) in enumerate(paired_inputs):
            data1 = self.embedding_data[text1]
            data2 = self.embedding_data[text2]

            # Get hidden states
            states1 = data1["hidden_states"]
            states2 = data2["hidden_states"]

            # Calculate distances for each layer
            for layer_idx in range(max_layers):
                if layer_idx < len(states1) and layer_idx < len(states2):
                    # Get mean representations
                    rep1 = states1[layer_idx].mean(dim=1).numpy()
                    rep2 = states2[layer_idx].mean(dim=1).numpy()

                    # Calculate distance
                    distance = np.linalg.norm(rep1 - rep2)

                    # Normalize by dimensionality
                    normalized_distance = distance / (rep1.shape[1] ** 0.5)

                    # Store distance
                    if data1.get("base_hidden_states") is not None and data2.get("base_hidden_states") is not None:
                        # This is a comparison between base and verified
                        base_distances[pair_idx, layer_idx] = normalized_distance
                    else:
                        # This is between verified states
                        verified_distances[pair_idx, layer_idx] = normalized_distance

        # Get adapter locations
        adapter_locations = self.embedding_data[all_texts[0]]["adapter_locations"]

        # Create distance evolution plot
        plt.figure(figsize=(12, 6))

        # Plot average distances across pairs
        base_avg = np.mean(base_distances, axis=0)
        verified_avg = np.mean(verified_distances, axis=0)

        # Ensure we don't plot zero values (where no data exists)
        base_avg_masked = np.ma.masked_where(base_avg == 0, base_avg)
        verified_avg_masked = np.ma.masked_where(verified_avg == 0, verified_avg)

        # Plot means
        plt.plot(range(max_layers), base_avg_masked, 'b-', label='Base Distance', linewidth=2)
        plt.plot(range(max_layers), verified_avg_masked, 'r-', label='Verified Distance', linewidth=2)

        # Add shaded regions for standard deviation
        if len(paired_inputs) > 1:
            base_std = np.std(base_distances, axis=0)
            verified_std = np.std(verified_distances, axis=0)

            plt.fill_between(
                range(max_layers),
                base_avg_masked - base_std,
                base_avg_masked + base_std,
                color='blue', alpha=0.2
            )

            plt.fill_between(
                range(max_layers),
                verified_avg_masked - verified_std,
                verified_avg_masked + verified_std,
                color='red', alpha=0.2
            )

        # Mark adapter layers
        for adapter_idx in adapter_locations:
            plt.axvline(x=adapter_idx, color='g', linestyle='--', alpha=0.5)

        plt.xlabel('Layer')
        plt.ylabel('Normalized Distance')
        plt.title('Distance Between Contradicting Statements Across Layers')
        plt.legend()
        plt.grid(alpha=0.3)

        # Save figure
        output_path = os.path.join(self.output_dir, "contrastive", "distance_evolution.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved embedding distance evolution plot to {output_path}")

        # Create heatmap of distances across all pairs and layers
        plt.figure(figsize=(12, 8))

        # Create distance difference heatmap (verified - base)
        distance_diff = verified_distances - base_distances

        # Mask zero values (no data)
        distance_diff_masked = np.ma.masked_where(
            (verified_distances == 0) | (base_distances == 0),
            distance_diff
        )

        # Create heatmap
        sns.heatmap(
            distance_diff_masked,
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            xticklabels=[f"Layer {i}" for i in range(max_layers)],
            yticklabels=[f"Pair {i+1}" for i in range(len(paired_inputs))]
        )

        # Mark adapter layers
        ax = plt.gca()
        for adapter_idx in adapter_locations:
            ax.add_patch(plt.Rectangle((adapter_idx, 0), 1, len(paired_inputs), 
                                   fill=False, edgecolor='green', lw=2))

        plt.title('Distance Difference (Verified - Base) Between Contradicting Statements')
        plt.tight_layout()

        # Save figure
        output_path = os.path.join(self.output_dir, "contrastive", "distance_heatmap.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved embedding distance heatmap to {output_path}")

        # Create bar chart of average distance change at adapter layers
        plt.figure(figsize=(10, 6))

        # Calculate average distance change at each adapter layer
        adapter_diffs = []

        for adapter_idx in adapter_locations:
            if adapter_idx < max_layers:
                # Get column from distance_diff
                col = distance_diff[:, adapter_idx]
                # Filter out zero values
                col_filtered = col[col != 0]

                if len(col_filtered) > 0:
                    adapter_diffs.append(np.mean(col_filtered))
                else:
                    adapter_diffs.append(0)

        # Create bar chart
        bars = plt.bar(
            range(len(adapter_locations)),
            adapter_diffs,
            color=['green' if d > 0 else 'red' for d in adapter_diffs]
        )

        # Add value labels
        for bar, diff in zip(bars, adapter_diffs):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01 if height > 0 else height - 0.03,
                f"{diff:.3f}",
                ha='center',
                va='bottom' if height > 0 else 'top',
                fontsize=10
            )

        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('Adapter Index')
        plt.ylabel('Average Distance Change')
        plt.title('Average Distance Change at Adapter Layers (Verified - Base)')
        plt.xticks(range(len(adapter_locations)), [f"Layer {idx}" for idx in adapter_locations])
        plt.grid(axis='y', alpha=0.3)

        # Save figure
        output_path = os.path.join(self.output_dir, "contrastive", "adapter_distance_change.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved adapter distance change plot to {output_path}")

    def analyze_attention_patterns(self, input_texts: List[str] = None):
        """
        Analyze attention patterns in verification layers

        Args:
            input_texts: List of inputs to analyze
                If None, default examples will be used
        """
        if self.verified_model is None:
            self.load_models()

        # Use default inputs if not provided
        if input_texts is None:
            input_texts = [
                "The capital of France is Berlin.",
                "The capital of France is Paris.",
                "The tallest mountain in the world is Mount Everest.",
                "The human heart has four chambers."
            ]

        logger.info(f"Analyzing attention patterns for {len(input_texts)} inputs...")

        # Process each input
        for i, text in enumerate(input_texts):
            # Skip if already processed
            if text in self.embedding_data and "attention_patterns" in self.embedding_data[text]:
                continue

            # Get embeddings and attention
            embedding_data = self._extract_embedding_dynamics(text)

            # Store data
            self.embedding_data[text] = embedding_data

            # Skip if no attention patterns
            if embedding_data["attention_patterns"] is None:
                continue

            # Visualize attention patterns
            self._visualize_attention_patterns(i, text, embedding_data)

        # Create aggregate attention comparison
        self._create_aggregate_attention_comparison(input_texts)

    def _visualize_attention_patterns(self, idx: int, text: str, embedding_data: Dict[str, Any]):
        """
        Visualize attention patterns for a given input

        Args:
            idx: Index of the input
            text: Input text
            embedding_data: Embedding dynamics data for the input
        """
        # Get attention patterns and adapter locations
        attention_patterns = embedding_data["attention_patterns"]
        adapter_locations = embedding_data["adapter_locations"]

        # Skip if no attention patterns
        if attention_patterns is None:
            return

        # Tokenize the input to get token labels
        tokens = self.tokenizer.tokenize(text)

        # Handle case of different number of tokens
        token_length = attention_patterns[0].shape[-1]
        if len(tokens) != token_length:
            # Truncate or pad tokens to match attention matrix size
            if len(tokens) > token_length:
                tokens = tokens[:token_length]
            else:
                tokens = tokens + ["[PAD]"] * (token_length - len(tokens))

        # Select layers to visualize (first, last, and adapters)
        key_layers = [0, len(attention_patterns) - 1]  # First and last
        key_layers.extend([min(i, len(attention_patterns) - 1) for i in adapter_locations])  # Adapters
        key_layers = sorted(set(key_layers))  # Remove duplicates and sort

        # Create figure with subplots for each key layer
        fig, axes = plt.subplots(
            1, len(key_layers),
            figsize=(4 * len(key_layers), 4),
            constrained_layout=True
        )

        # Handle case of single subplot
        if len(key_layers) == 1:
            axes = [axes]

        # Process each key layer
        for i, layer_idx in enumerate(key_layers):
            ax = axes[i]

            # Get attention pattern for this layer
            # Average attention across all heads
            attn = attention_patterns[layer_idx].mean(dim=1).squeeze().numpy()

            # Create heatmap
            im = ax.imshow(attn, cmap="viridis")

            # Set title
            if layer_idx in adapter_locations:
                ax.set_title(f"Layer {layer_idx}\n(Verification)")
            else:
                ax.set_title(f"Layer {layer_idx}")

            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)

            # Set tick labels
            if token_length <= 10:
                # If few tokens, show all labels
                ax.set_xticks(range(token_length))
                ax.set_yticks(range(token_length))
                ax.set_xticklabels(tokens, rotation=45, ha="right")
                ax.set_yticklabels(tokens)
            else:
                # If many tokens, show fewer tick labels
                step = max(1, token_length // 10)
                ax.set_xticks(range(0, token_length, step))
                ax.set_yticks(range(0, token_length, step))
                ax.set_xticklabels([tokens[i] for i in range(0, token_length, step)], rotation=45, ha="right")
                ax.set_yticklabels([tokens[i] for i in range(0, token_length, step)])

        plt.suptitle(f"Attention Patterns: {text}", fontsize=16)

        # Save figure
        output_path = os.path.join(self.output_dir, "attention", f"attention_{idx+1}.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved attention patterns to {output_path}")

    def _create_aggregate_attention_comparison(self, input_texts: List[str]):
        """
        Create aggregate attention pattern comparison across inputs

        Args:
            input_texts: List of input texts
        """
        # Filter to inputs with attention patterns
        valid_texts = [text for text in input_texts if 
                    text in self.embedding_data and 
                    self.embedding_data[text]["attention_patterns"] is not None]

        # Skip if fewer than 2 valid texts
        if len(valid_texts) < 2:
            return

        # Get adapter locations from first text
        adapter_locations = self.embedding_data[valid_texts[0]]["adapter_locations"]

        # Skip if no adapter locations
        if not adapter_locations:
            return

        # Choose a middle adapter layer for comparison
        middle_adapter_idx = adapter_locations[len(adapter_locations) // 2]

        # Create figure with subplots for each input
        fig, axes = plt.subplots(
            1, len(valid_texts),
            figsize=(4 * len(valid_texts), 4),
            constrained_layout=True
        )

        # Handle case of single subplot
        if len(valid_texts) == 1:
            axes = [axes]

        # Create shared colormap normalization
        all_attns = []
        for text in valid_texts:
            data = self.embedding_data[text]
            attention_patterns = data["attention_patterns"]

            if middle_adapter_idx < len(attention_patterns):
                attn = attention_patterns[middle_adapter_idx].mean(dim=1).squeeze().numpy()
                all_attns.append(attn)

        if not all_attns:
            return

        # Calculate global min and max for colormap normalization
        global_min = min(attn.min() for attn in all_attns)
        global_max = max(attn.max() for attn in all_attns)
        norm = colors.Normalize(vmin=global_min, vmax=global_max)

        # Process each input
        for i, text in enumerate(valid_texts):
            ax = axes[i]
            data = self.embedding_data[text]
            attention_patterns = data["attention_patterns"]

            if middle_adapter_idx < len(attention_patterns):
                attn = attention_patterns[middle_adapter_idx].mean(dim=1).squeeze().numpy()

                # Tokenize to get labels
                tokens = self.tokenizer.tokenize(text)

                # Handle case of different number of tokens
                token_length = attn.shape[0]
                if len(tokens) != token_length:
                    # Truncate or pad tokens to match attention matrix size
                    if len(tokens) > token_length:
                        tokens = tokens[:token_length]
                    else:
                        tokens = tokens + ["[PAD]"] * (token_length - len(tokens))

                # Create heatmap
                im = ax.imshow(attn, cmap="viridis", norm=norm)

                # Set title
                ax.set_title(f"{text[:20]}..." if len(text) > 20 else text)

                # Add colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(im, cax=cax)

                # Set tick labels
                if token_length <= 10:
                    # If few tokens, show all labels
                    ax.set_xticks(range(token_length))
                    ax.set_yticks(range(token_length))
                    ax.set_xticklabels(tokens, rotation=45, ha="right")
                    ax.set_yticklabels(tokens)
                else:
                    # If many tokens, show fewer tick labels
                    step = max(1, token_length // 10)
                    ax.set_xticks(range(0, token_length, step))
                    ax.set_yticks(range(0, token_length, step))
                    ax.set_xticklabels([tokens[i] for i in range(0, token_length, step)], rotation=45, ha="right")
                    ax.set_yticklabels([tokens[i] for i in range(0, token_length, step)])

        plt.suptitle(f"Attention Patterns at Layer {middle_adapter_idx} (Verification)", fontsize=16)

        # Save figure
        output_path = os.path.join(self.output_dir, "attention", "aggregate_attention_comparison.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved aggregate attention comparison to {output_path}")

    def analyze_contrastive_representation(self, input_texts: List[str] = None):
        """
        Analyze how verification changes the relationship between contradictory statements

        Args:
            input_texts: List of inputs to analyze
                If None, default examples will be used
        """
        if self.verified_model is None or self.base_model is None:
            if self.base_model is None:
                logger.warning("Base model not available. Skipping contrastive representation analysis.")
                return
            self.load_models()

        # Use default inputs if not provided
        if input_texts is None:
            input_texts = [
                "The capital of France is Berlin.",
                "The capital of France is Paris.",
                "The tallest mountain in the world is K2.",
                "The tallest mountain in the world is Mount Everest.",
                "Water boils at 50 degrees Celsius at sea level.",
                "Water boils at 100 degrees Celsius at sea level.",
                "The human heart has three chambers.",
                "The human heart has four chambers.",
                "The Earth orbits around the Sun.",
                "The Sun orbits around the Earth."
            ]

        logger.info(f"Analyzing contrastive representation for {len(input_texts)} inputs...")

        # Group inputs into pairs of contradicting statements
        paired_inputs = []
        for i in range(0, len(input_texts), 2):
            if i + 1 < len(input_texts):
                paired_inputs.append((input_texts[i], input_texts[i+1]))

        # Process each pair
        for pair_idx, (text1, text2) in enumerate(paired_inputs):
            # Get embeddings for text1 if not already processed
            if text1 not in self.embedding_data:
                embedding_data1 = self._extract_embedding_dynamics(text1)
                self.embedding_data[text1] = embedding_data1
            else:
                embedding_data1 = self.embedding_data[text1]

            # Get embeddings for text2 if not already processed
            if text2 not in self.embedding_data:
                embedding_data2 = self._extract_embedding_dynamics(text2)
                self.embedding_data[text2] = embedding_data2
            else:
                embedding_data2 = self.embedding_data[text2]

            # Visualize contrastive representation
            self._visualize_contrastive_representation(pair_idx, text1, text2, embedding_data1, embedding_data2)

        # Create aggregate contrastive analysis
        self._create_aggregate_contrastive_analysis(paired_inputs)

    def _visualize_contrastive_representation(self, pair_idx: int, text1: str, text2: str, 
                                           embedding_data1: Dict[str, Any], embedding_data2: Dict[str, Any]):
        """
        Visualize contrastive representation for a pair of contradicting statements

        Args:
            pair_idx: Index of the pair
            text1: First input text
            text2: Second input text
            embedding_data1: Embedding dynamics data for text1
            embedding_data2: Embedding dynamics data for text2
        """
        # Get adapter locations
        adapter_locations = embedding_data1["adapter_locations"]

        # Skip if no adapter locations
        if not adapter_locations:
            return

        # Get hidden states
        base_states1 = embedding_data1.get("base_hidden_states")
        base_states2 = embedding_data2.get("base_hidden_states")
        verified_states1 = embedding_data1["hidden_states"]
        verified_states2 = embedding_data2["hidden_states"]

        # Skip if any states are missing
        if base_states1 is None or base_states2 is None:
            logger.warning("Base model hidden states not available. Skipping contrastive visualization.")
            return

        # Calculate cosine similarities between the pair at each layer
        base_similarities = []
        verified_similarities = []

        min_layers = min(len(base_states1), len(base_states2), len(verified_states1), len(verified_states2))

        for layer_idx in range(min_layers):
            # Calculate base similarity
            base_repr1 = base_states1[layer_idx].mean(dim=1).squeeze()
            base_repr2 = base_states2[layer_idx].mean(dim=1).squeeze()

            base_sim = F.cosine_similarity(base_repr1, base_repr2, dim=0).item()
            base_similarities.append(base_sim)

            # Calculate verified similarity
            verified_repr1 = verified_states1[layer_idx].mean(dim=1).squeeze()
            verified_repr2 = verified_states2[layer_idx].mean(dim=1).squeeze()

            verified_sim = F.cosine_similarity(verified_repr1, verified_repr2, dim=0).item()
            verified_similarities.append(verified_sim)

        # Create similarity plot
        plt.figure(figsize=(12, 6))

        # Plot similarities
        plt.plot(range(min_layers), base_similarities, 'b-', linewidth=2, label='Base Model')
        plt.plot(range(min_layers), verified_similarities, 'r-', linewidth=2, label='Verified Model')

        # Mark adapter layers
        for adapter_idx in adapter_locations:
            if adapter_idx < min_layers:
                plt.axvline(x=adapter_idx, color='g', linestyle='--', alpha=0.5)

        plt.xlabel('Layer')
        plt.ylabel('Cosine Similarity')
        plt.title(f'Similarity Between "{text1[:20]}..." and "{text2[:20]}..."')
        plt.legend()
        plt.grid(alpha=0.3)

        # Save figure
        output_path = os.path.join(self.output_dir, "contrastive", f"similarity_{pair_idx+1}.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved contrastive similarity plot to {output_path}")

    def _create_aggregate_contrastive_analysis(self, paired_inputs: List[Tuple[str, str]]):
        """
        Create aggregate contrastive analysis across pairs

        Args:
            paired_inputs: List of (text1, text2) pairs
        """
        # Skip if no pairs
        if not paired_inputs:
            return

        # Check if base model states are available
        has_base_states = True
        for text1, text2 in paired_inputs:
            if (text1 not in self.embedding_data or 
                text2 not in self.embedding_data or 
                self.embedding_data[text1].get("base_hidden_states") is None or 
                self.embedding_data[text2].get("base_hidden_states") is None):
                has_base_states = False
                break

        if not has_base_states:
            logger.warning("Base model hidden states not available for some pairs. Skipping aggregate contrastive analysis.")
            return

        # Get adapter locations
        adapter_locations = self.embedding_data[paired_inputs[0][0]]["adapter_locations"]

        # Calculate similarities for all pairs
        base_similarities = []
        verified_similarities = []

        for text1, text2 in paired_inputs:
            embedding_data1 = self.embedding_data[text1]
            embedding_data2 = self.embedding_data[text2]

            base_states1 = embedding_data1["base_hidden_states"]
            base_states2 = embedding_data2["base_hidden_states"]
            verified_states1 = embedding_data1["hidden_states"]
            verified_states2 = embedding_data2["hidden_states"]

            min_layers = min(len(base_states1), len(base_states2), len(verified_states1), len(verified_states2))

            pair_base_sims = []
            pair_verified_sims = []

            for layer_idx in range(min_layers):
                # Calculate base similarity
                base_repr1 = base_states1[layer_idx].mean(dim=1).squeeze()
                base_repr2 = base_states2[layer_idx].mean(dim=1).squeeze()

                base_sim = F.cosine_similarity(base_repr1, base_repr2, dim=0).item()
                pair_base_sims.append(base_sim)

                # Calculate verified similarity
                verified_repr1 = verified_states1[layer_idx].mean(dim=1).squeeze()
                verified_repr2 = verified_states2[layer_idx].mean(dim=1).squeeze()

                verified_sim = F.cosine_similarity(verified_repr1, verified_repr2, dim=0).item()
                pair_verified_sims.append(verified_sim)

            base_similarities.append(pair_base_sims)
            verified_similarities.append(pair_verified_sims)

        # Calculate average similarities
        max_layers = max(len(sims) for sims in base_similarities + verified_similarities)

        base_avg = np.zeros(max_layers)
        verified_avg = np.zeros(max_layers)
        base_count = np.zeros(max_layers)
        verified_count = np.zeros(max_layers)

        for sims in base_similarities:
            for i, sim in enumerate(sims):
                base_avg[i] += sim
                base_count[i] += 1

        for sims in verified_similarities:
            for i, sim in enumerate(sims):
                verified_avg[i] += sim
                verified_count[i] += 1

        # Avoid division by zero
        base_count[base_count == 0] = 1
        verified_count[verified_count == 0] = 1

        base_avg /= base_count
        verified_avg /= verified_count

        # Create aggregate similarity plot
        plt.figure(figsize=(12, 6))

        # Plot average similarities
        plt.plot(range(max_layers), base_avg, 'b-', linewidth=2, label='Base Model')
        plt.plot(range(max_layers), verified_avg, 'r-', linewidth=2, label='Verified Model')

        # Calculate and plot standard deviation if multiple pairs
        if len(paired_inputs) > 1:
            base_std = np.zeros(max_layers)
            verified_std = np.zeros(max_layers)

            for sims in base_similarities:
                for i, sim in enumerate(sims):
                    base_std[i] += (sim - base_avg[i]) ** 2

            for sims in verified_similarities:
                for i, sim in enumerate(sims):
                    verified_std[i] += (sim - verified_avg[i]) ** 2

            base_std = np.sqrt(base_std / base_count)
            verified_std = np.sqrt(verified_std / verified_count)

            plt.fill_between(
                range(max_layers),
                base_avg - base_std,
                base_avg + base_std,
                color='blue', alpha=0.2
            )

            plt.fill_between(
                range(max_layers),
                verified_avg - verified_std,
                verified_avg + verified_std,
                color='red', alpha=0.2
            )

        # Mark adapter layers
        for adapter_idx in adapter_locations:
            if adapter_idx < max_layers:
                plt.axvline(x=adapter_idx, color='g', linestyle='--', alpha=0.5)

        plt.xlabel('Layer')
        plt.ylabel('Average Cosine Similarity')
        plt.title('Average Similarity Between Contradicting Statements')
        plt.legend()
        plt.grid(alpha=0.3)

        # Save figure
        output_path = os.path.join(self.output_dir, "contrastive", "aggregate_similarity.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved aggregate contrastive similarity plot to {output_path}")

        # Create similarity difference plot
        plt.figure(figsize=(12, 6))

        # Calculate and plot similarity differences (verified - base)
        sim_diff = verified_avg - base_avg

        bars = plt.bar(
            range(max_layers),
            sim_diff,
            color=['green' if d < 0 else 'red' for d in sim_diff]
        )

        # Add value labels
        for bar, diff in zip(bars, sim_diff):
            height = bar.get_height()

            # Skip bars with no height
            if height == 0:
                continue

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01 if height > 0 else height - 0.03,
                f"{diff:.3f}",
                ha='center',
                va='bottom' if height > 0 else 'top',
                fontsize=10
            )

        # Mark adapter layers
        for adapter_idx in adapter_locations:
            if adapter_idx < max_layers:
                plt.axvline(x=adapter_idx, color='g', linestyle='--', alpha=0.5)

        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('Layer')
        plt.ylabel('Similarity Difference (Verified - Base)')
        plt.title('How Verification Changes Similarity Between Contradicting Statements')
        plt.grid(axis='y', alpha=0.3)

        # Save figure
        output_path = os.path.join(self.output_dir, "contrastive", "similarity_difference.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved similarity difference plot to {output_path}")

def main():
    """Run the embedding dynamics analyzer from command line"""
    parser = argparse.ArgumentParser(description="Analyze embedding dynamics in latent verification")
    parser.add_argument("--verified_model", type=str, required=True, help="Path to verification-enhanced model")
    parser.add_argument("--base_model", type=str, default=None, help="Base model name (optional)")
    parser.add_argument("--output_dir", type=str, default="embedding_dynamics", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--high_quality", action="store_true", help="Generate high-quality (but slower) visualizations")

    parser.add_argument("--run_all", action="store_true", help="Run all analyses")
    parser.add_argument("--analyze_embeddings", action="store_true", help="Run embedding dynamics analysis")
    parser.add_argument("--analyze_attention", action="store_true", help="Run attention pattern analysis")
    parser.add_argument("--analyze_contrastive", action="store_true", help="Run contrastive representation analysis")

    args = parser.parse_args()

    # Create analyzer
    analyzer = EmbeddingDynamicsAnalyzer(
        verified_model_path=args.verified_model,
        base_model_name=args.base_model,
        output_dir=args.output_dir,
        device=args.device,
        high_quality=args.high_quality
    )

    # Determine which analyses to run
    run_embeddings = args.run_all or args.analyze_embeddings
    run_attention = args.run_all or args.analyze_attention
    run_contrastive = args.run_all or args.analyze_contrastive

    # If no specific analyses are selected, run embedding dynamics by default
    if not (run_embeddings or run_attention or run_contrastive or run_report):
        run_embeddings = True
        run_report = True

    # Run selected analyses
    if run_embeddings:
        analyzer.analyze_embedding_dynamics()

    if run_attention:
        analyzer.analyze_attention_patterns()

    if run_contrastive:
        analyzer.analyze_contrastive_representation()

if __name__ == "__main__":
    main()
