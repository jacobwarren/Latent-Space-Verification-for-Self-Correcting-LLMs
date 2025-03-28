#!/usr/bin/env python
"""
Latent Thinking Analysis Tool

This script deeply analyzes and visualizes the "thinking in latent space" 
capabilities of verification-enhanced language models. It provides comprehensive
analysis focusing on:

1. Hidden state trajectory analysis - How hidden states evolve through layers
2. Verification-induced corrections - How verification adapters modify representations
3. Truth vs. falsehood divergence - How corrections differ based on truthfulness
4. Token probability flow analysis - How latent corrections influence output probabilities
5. Comparative trajectory visualization - Side-by-side contrast with base model

This is the most specialized tool for the direct measurement and visualization
of the latent verification mechanism's impact on the model's internal representations.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Union
import json
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle, Arrow
from matplotlib.collections import PatchCollection

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("latent_thinking.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LatentThinkingAnalyzer:
    """
    Specialized analyzer for studying "thinking in latent space" through
    verification mechanisms.
    """
    def __init__(
        self,
        base_model_name: str,
        verified_model_path: str,
        output_dir: str = "latent_thinking_analysis",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the analyzer with model paths

        Args:
            base_model_name: Name or path of the original base model
            verified_model_path: Path to the model with verification components
            output_dir: Directory to save analysis results
            device: Device to run analysis on
        """
        self.base_model_name = base_model_name
        self.verified_model_path = verified_model_path
        self.output_dir = output_dir
        self.device = device

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize models
        self.tokenizer = None
        self.base_model = None
        self.verified_model = None

        # Analysis results
        self.results = {}

        logger.info(f"Initializing analyzer with base model: {base_model_name}")
        logger.info(f"Verified model path: {verified_model_path}")
        logger.info(f"Device: {device}")

    def load_models(self):
        """Load models for analysis"""
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        logger.info("Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name
        ).to(self.device)

        # Load verified model
        logger.info("Loading verified model...")

        try:
            # First try loading with verification wrapper
            from enhanced_verification import load_bayesian_verification_model
            self.verified_model = load_bayesian_verification_model(self.verified_model_path).to(self.device)
            logger.info("Successfully loaded verified model with wrapper")
        except Exception as e:
            logger.warning(f"Error loading with wrapper: {e}. Trying standard loading...")
            # Fallback to standard loading
            self.verified_model = AutoModelForCausalLM.from_pretrained(
                self.verified_model_path
            ).to(self.device)
            logger.info("Loaded verified model with standard loading")

    def prepare_test_examples(self):
        """Prepare test examples for latent thinking analysis"""
        # Truth/falsehood statement pairs
        truth_falsehood_pairs = [
            # Factual knowledge
            ("The capital of France is Paris.", "The capital of France is Berlin."),
            ("The Earth orbits around the Sun.", "The Sun orbits around the Earth."),
            ("Water boils at 100 degrees Celsius at sea level.", "Water boils at 50 degrees Celsius at sea level."),
            ("The human heart has four chambers.", "The human heart has three chambers."),

            # Mathematical statements
            ("Two plus two equals four.", "Two plus two equals five."),
            ("The square root of 16 is 4.", "The square root of 16 is 5."),

            # Logical consistency
            ("All mammals are animals. Dogs are mammals. Therefore, dogs are animals.",
             "All mammals are animals. Dogs are mammals. Therefore, dogs are plants."),
        ]

        # Complex reasoning examples
        reasoning_examples = [
            "If all A are B, and all B are C, then all A are C. All dogs are mammals. All mammals are animals. Therefore, all dogs are",
            "The sum of the angles in a triangle is 180 degrees. If two angles in a triangle are 45 degrees and 45 degrees, then the third angle must be",
            "If it's raining, the ground gets wet. The ground is wet. Therefore, it",
            "If I add 5 to a number and then multiply by 2, I get 18. The original number must be",
        ]

        # Ambiguous statements that require verification
        ambiguous_examples = [
            "The Eiffel Tower is taller than the Statue of Liberty by approximately",
            "The distance from Earth to the Moon is about",
            "The most populous country in the world is",
            "The inventor of the telephone was",
        ]

        return {
            "truth_falsehood_pairs": truth_falsehood_pairs,
            "reasoning_examples": reasoning_examples,
            "ambiguous_examples": ambiguous_examples
        }

    def analyze_hidden_state_trajectories(self, prompts=None, num_layers=None):
        """
        Analyze how hidden states evolve through the layers,
        comparing base model vs. verification model

        Args:
            prompts: List of prompts to analyze
            num_layers: Number of layers to include in analysis
        """
        if self.base_model is None or self.verified_model is None:
            self.load_models()

        logger.info("Analyzing hidden state trajectories...")

        # Use default prompts if none provided
        if prompts is None:
            examples = self.prepare_test_examples()
            # Use first 3 truth/falsehood pairs
            prompts = []
            for pair in examples["truth_falsehood_pairs"][:3]:
                prompts.extend(pair)

        # Determine number of layers
        if num_layers is None:
            # Get from config
            num_layers = getattr(self.base_model.config, 'num_hidden_layers', 
                               getattr(self.base_model.config, 'n_layer', 
                                      getattr(self.base_model.config, 'num_layers', 12)))

        # Results storage
        trajectory_results = {
            "prompts": prompts,
            "base_trajectories": [],
            "verified_trajectories": [],
            "layer_distances_base": [],
            "layer_distances_verified": [], 
            "adapter_locations": [],
            "adapter_impacts": []
        }

        # Get adapter locations if available
        if hasattr(self.verified_model, 'adapter_locations'):
            trajectory_results["adapter_locations"] = self.verified_model.adapter_locations

        # Process each prompt
        for prompt in prompts:
            logger.info(f"Processing prompt: {prompt}")

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Process with base model
            with torch.no_grad():
                base_outputs = self.base_model(**inputs, output_hidden_states=True)
                base_hidden_states = [h.detach().cpu() for h in base_outputs.hidden_states]

            # Process with verified model
            with torch.no_grad():
                # Store original hooks
                old_all_adapter_hidden_states = getattr(self.verified_model, 'all_adapter_hidden_states', [])
                old_all_confidence_scores = getattr(self.verified_model, 'all_confidence_scores', [])

                # Reset collections
                if hasattr(self.verified_model, 'all_adapter_hidden_states'):
                    self.verified_model.all_adapter_hidden_states = []
                if hasattr(self.verified_model, 'all_confidence_scores'):
                    self.verified_model.all_confidence_scores = []

                verified_outputs = self.verified_model(**inputs, output_hidden_states=True)
                verified_hidden_states = [h.detach().cpu() for h in verified_outputs.hidden_states]

                # Get adapter hidden states if available
                adapter_states = getattr(self.verified_model, 'all_adapter_hidden_states', [])
                adapter_impacts = []

                # Calculate adapter impacts
                adapter_locations = trajectory_results["adapter_locations"]
                if adapter_states and adapter_locations:
                    for i, (adapter_idx, adapter_state) in enumerate(zip(adapter_locations, adapter_states)):
                        if adapter_idx < len(verified_hidden_states):
                            # Calculate impact as L2 norm of difference
                            orig_state = verified_hidden_states[adapter_idx]
                            diff = (adapter_state.cpu() - orig_state).norm().item()
                            # Normalize by tensor size
                            norm_diff = diff / (orig_state.numel() ** 0.5)
                            adapter_impacts.append(norm_diff)

                # Restore original hooks
                if hasattr(self.verified_model, 'all_adapter_hidden_states'):
                    self.verified_model.all_adapter_hidden_states = old_all_adapter_hidden_states
                if hasattr(self.verified_model, 'all_confidence_scores'):
                    self.verified_model.all_confidence_scores = old_all_confidence_scores

            # Calculate layer-to-layer distances for base model
            base_distances = []
            for i in range(1, len(base_hidden_states)):
                prev = base_hidden_states[i-1]
                curr = base_hidden_states[i]

                # Calculate L2 distance normalized by size
                dist = (curr - prev).norm().item() / (prev.numel() ** 0.5)
                base_distances.append(dist)

            # Calculate layer-to-layer distances for verified model
            verified_distances = []
            for i in range(1, len(verified_hidden_states)):
                prev = verified_hidden_states[i-1]
                curr = verified_hidden_states[i]

                # Calculate L2 distance normalized by size
                dist = (curr - prev).norm().item() / (prev.numel() ** 0.5)
                verified_distances.append(dist)

            # Store results
            trajectory_results["base_trajectories"].append(base_hidden_states)
            trajectory_results["verified_trajectories"].append(verified_hidden_states)
            trajectory_results["layer_distances_base"].append(base_distances)
            trajectory_results["layer_distances_verified"].append(verified_distances)
            trajectory_results["adapter_impacts"].append(adapter_impacts)

        # Save results
        self.results["hidden_state_trajectories"] = trajectory_results

        # Generate visualizations
        self._visualize_trajectories(trajectory_results)

        return trajectory_results

    def _visualize_trajectories(self, trajectory_results):
        """Create visualizations of hidden state trajectories"""
        prompts = trajectory_results["prompts"]
        base_distances = trajectory_results["layer_distances_base"]
        verified_distances = trajectory_results["layer_distances_verified"]
        adapter_locations = trajectory_results["adapter_locations"]
        adapter_impacts = trajectory_results["adapter_impacts"]

        # 1. Plot layer-wise distances for each prompt
        for i, (prompt, base_dist, verified_dist) in enumerate(zip(prompts, base_distances, verified_distances)):
            plt.figure(figsize=(12, 6))

            # Plot distances
            plt.plot(range(len(base_dist)), base_dist, 'b-', linewidth=2, label='Base Model')
            plt.plot(range(len(verified_dist)), verified_dist, 'r-', linewidth=2, label='Verified Model')

            # Add markers for adapter locations
            if adapter_locations:
                for loc in adapter_locations:
                    if loc < len(verified_dist):
                        plt.axvline(x=loc, color='g', linestyle='--', alpha=0.5)

            plt.xlabel('Layer')
            plt.ylabel('Normalized Hidden State Change')

            # Abbreviate long prompts
            if len(prompt) > 50:
                display_prompt = prompt[:47] + "..."
            else:
                display_prompt = prompt

            plt.title(f'Hidden State Trajectory: "{display_prompt}"')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"trajectory_{i+1}.png"))
            plt.close()

        # 2. Plot adapter impacts if available
        if adapter_impacts and adapter_impacts[0]:
            plt.figure(figsize=(12, 6))

            # Plot impacts for each prompt
            for i, (prompt, impacts) in enumerate(zip(prompts, adapter_impacts)):
                if len(prompts) <= 6:  # Only label if few prompts
                    if len(prompt) > 20:
                        label = prompt[:17] + "..."
                    else:
                        label = prompt
                else:
                    label = f"Prompt {i+1}"

                plt.plot(range(len(impacts)), impacts, 'o-', linewidth=2, label=label)

            plt.xlabel('Adapter Index')
            plt.ylabel('Adapter Impact (Normalized)')
            plt.title('Verification Adapter Impacts Across Layers')

            if len(prompts) <= 6:  # Only show legend if few prompts
                plt.legend()

            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "adapter_impacts.png"))
            plt.close()

        # 3. Create a heatmap of distances
        if len(base_distances) >= 2:
            # Prepare data for heatmaps
            base_array = np.array(base_distances)
            verified_array = np.array(verified_distances)

            # Plot base model heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(base_array, cmap="viridis", 
                       xticklabels=[f"Layer {i}" for i in range(1, base_array.shape[1]+1)],
                       yticklabels=[f"Prompt {i+1}" for i in range(base_array.shape[0])])
            plt.title('Base Model: Hidden State Changes Between Layers')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "base_layer_distances_heatmap.png"))
            plt.close()

            # Plot verified model heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(verified_array, cmap="viridis", 
                       xticklabels=[f"Layer {i}" for i in range(1, verified_array.shape[1]+1)],
                       yticklabels=[f"Prompt {i+1}" for i in range(verified_array.shape[0])])

            # Highlight adapter locations
            if adapter_locations:
                ax = plt.gca()
                for loc in adapter_locations:
                    if loc < verified_array.shape[1]:
                        # Add a rectangle around this column
                        rect = plt.Rectangle((loc, 0), 1, verified_array.shape[0], 
                                          linewidth=2, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)

            plt.title('Verified Model: Hidden State Changes Between Layers')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "verified_layer_distances_heatmap.png"))
            plt.close()

            # Plot difference heatmap
            if base_array.shape == verified_array.shape:
                diff_array = verified_array - base_array
                plt.figure(figsize=(12, 8))
                sns.heatmap(diff_array, cmap="coolwarm", center=0,
                           xticklabels=[f"Layer {i}" for i in range(1, diff_array.shape[1]+1)],
                           yticklabels=[f"Prompt {i+1}" for i in range(diff_array.shape[0])])

                # Highlight adapter locations
                if adapter_locations:
                    ax = plt.gca()
                    for loc in adapter_locations:
                        if loc < diff_array.shape[1]:
                            # Add a rectangle around this column
                            rect = plt.Rectangle((loc, 0), 1, diff_array.shape[0], 
                                              linewidth=2, edgecolor='r', facecolor='none')
                            ax.add_patch(rect)

                plt.title('Difference: Verified - Base Model Hidden State Changes')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, "difference_layer_distances_heatmap.png"))
                plt.close()

        # 4. Create 2D PCA visualization of hidden states evolution
        self._visualize_hidden_state_pca(trajectory_results)

    def _visualize_hidden_state_pca(self, trajectory_results):
        """Visualize evolution of hidden states in 2D using PCA"""
        # Only visualize first prompt for clarity
        base_states = trajectory_results["base_trajectories"][0]
        verified_states = trajectory_results["verified_trajectories"][0]
        prompt = trajectory_results["prompts"][0]
        adapter_locations = trajectory_results["adapter_locations"]

        # Extract [CLS] or first token representation from each layer
        base_tokens = [states[:, 0, :].numpy() for states in base_states]
        verified_tokens = [states[:, 0, :].numpy() for states in verified_states]

        # Combine all states for PCA
        all_states = np.vstack(base_tokens + verified_tokens)

        # Apply PCA
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(all_states)

        # Split back into base and verified
        n_layers = len(base_tokens)
        base_reduced = reduced[:n_layers]
        verified_reduced = reduced[n_layers:]

        # Create plot
        plt.figure(figsize=(12, 10))

        # Plot base model trajectory
        plt.plot(base_reduced[:, 0], base_reduced[:, 1], 'b-', linewidth=2, 
                alpha=0.7, label='Base Model')

        # Add points for each layer
        for i, (x, y) in enumerate(base_reduced):
            plt.scatter(x, y, color='blue', s=100, alpha=0.7)
            plt.text(x, y, str(i), color='white', ha='center', va='center', fontsize=8)

        # Plot verified model trajectory
        plt.plot(verified_reduced[:, 0], verified_reduced[:, 1], 'r-', linewidth=2, 
                alpha=0.7, label='Verified Model')

        # Add points for each layer
        for i, (x, y) in enumerate(verified_reduced):
            # Use different marker for adapter layers
            if i in adapter_locations:
                plt.scatter(x, y, color='red', s=150, alpha=0.7, marker='*')
            else:
                plt.scatter(x, y, color='red', s=100, alpha=0.7)
            plt.text(x, y, str(i), color='white', ha='center', va='center', fontsize=8)

        plt.title(f'Hidden State Evolution in 2D (PCA)\nPrompt: "{prompt[:50]}..."')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "hidden_state_pca.png"))
        plt.close()

        # Create animation of the trajectory
        fig, ax = plt.subplots(figsize=(12, 10))

        def update(frame):
            ax.clear()

            # Plot base trajectory up to current frame
            base_x = base_reduced[:frame+1, 0]
            base_y = base_reduced[:frame+1, 1]
            ax.plot(base_x, base_y, 'b-', linewidth=2, alpha=0.7, label='Base Model')

            # Plot verified trajectory up to current frame
            verified_x = verified_reduced[:frame+1, 0]
            verified_y = verified_reduced[:frame+1, 1]
            ax.plot(verified_x, verified_y, 'r-', linewidth=2, alpha=0.7, label='Verified Model')

            # Add points for all layers
            for i in range(frame + 1):
                # Base model point
                ax.scatter(base_reduced[i, 0], base_reduced[i, 1], color='blue', s=100, alpha=0.7)
                ax.text(base_reduced[i, 0], base_reduced[i, 1], str(i), color='white', 
                       ha='center', va='center', fontsize=8)

                # Verified model point
                if i in adapter_locations:
                    ax.scatter(verified_reduced[i, 0], verified_reduced[i, 1], 
                              color='red', s=150, alpha=0.7, marker='*')
                else:
                    ax.scatter(verified_reduced[i, 0], verified_reduced[i, 1], 
                              color='red', s=100, alpha=0.7)
                ax.text(verified_reduced[i, 0], verified_reduced[i, 1], str(i), 
                       color='white', ha='center', va='center', fontsize=8)

            # Add connecting line between current layer in both models
            if frame < n_layers:
                ax.plot([base_reduced[frame, 0], verified_reduced[frame, 0]], 
                       [base_reduced[frame, 1], verified_reduced[frame, 1]], 
                       'g--', alpha=0.5)

            ax.set_title(f'Hidden State Evolution (Layer {frame})\nPrompt: "{prompt[:50]}..."')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)

            return ax

        # Create animation
        anim = FuncAnimation(fig, update, frames=n_layers, interval=500)
        anim.save(os.path.join(self.output_dir, 'hidden_state_evolution.gif'), 
                 writer='pillow', fps=2)
        plt.close()

    def analyze_truth_falsehood_divergence(self):
        """
        Analyze how verification mechanisms cause representations to
        diverge differently for true vs. false statements
        """
        if self.base_model is None or self.verified_model is None:
            self.load_models()

        logger.info("Analyzing truth/falsehood divergence...")

        # Get truth/falsehood pairs
        examples = self.prepare_test_examples()
        pairs = examples["truth_falsehood_pairs"]

        # Results storage
        divergence_results = {
            "pairs": pairs,
            "base_similarities": [],
            "verified_similarities": [],
            "base_divergence_layers": [],
            "verified_divergence_layers": [],
            "verification_magnitudes": []
        }

        # Process each true/false pair
        for i, (truth, falsehood) in enumerate(pairs):
            logger.info(f"Processing pair {i+1}: '{truth}' vs '{falsehood}'")

            # Tokenize inputs
            truth_inputs = self.tokenizer(truth, return_tensors="pt").to(self.device)
            falsehood_inputs = self.tokenizer(falsehood, return_tensors="pt").to(self.device)

            # Get hidden states from base model
            with torch.no_grad():
                truth_base = self.base_model(**truth_inputs, output_hidden_states=True)
                falsehood_base = self.base_model(**falsehood_inputs, output_hidden_states=True)

                # Extract hidden states
                truth_base_states = [h.detach().cpu() for h in truth_base.hidden_states]
                falsehood_base_states = [h.detach().cpu() for h in falsehood_base.hidden_states]

            # Get hidden states from verified model
            with torch.no_grad():
                # Reset adapter states
                if hasattr(self.verified_model, 'all_adapter_hidden_states'):
                    old_all_adapter_hidden_states = self.verified_model.all_adapter_hidden_states
                    self.verified_model.all_adapter_hidden_states = []

                if hasattr(self.verified_model, 'all_confidence_scores'):
                    old_all_confidence_scores = self.verified_model.all_confidence_scores
                    self.verified_model.all_confidence_scores = []

                truth_verified = self.verified_model(**truth_inputs, output_hidden_states=True)

                # Get truth confidence scores
                truth_confidence = []
                if hasattr(self.verified_model, 'all_confidence_scores'):
                    truth_confidence = [score.mean().item() for score in self.verified_model.all_confidence_scores]

                # Reset adapter states again
                if hasattr(self.verified_model, 'all_adapter_hidden_states'):
                    self.verified_model.all_adapter_hidden_states = []

                if hasattr(self.verified_model, 'all_confidence_scores'):
                    self.verified_model.all_confidence_scores = []

                falsehood_verified = self.verified_model(**falsehood_inputs, output_hidden_states=True)

                # Get falsehood confidence scores
                falsehood_confidence = []
                if hasattr(self.verified_model, 'all_confidence_scores'):
                    falsehood_confidence = [score.mean().item() for score in self.verified_model.all_confidence_scores]

                # Extract hidden states
                truth_verified_states = [h.detach().cpu() for h in truth_verified.hidden_states]
                falsehood_verified_states = [h.detach().cpu() for h in falsehood_verified.hidden_states]

                # Restore adapter states
                if hasattr(self.verified_model, 'all_adapter_hidden_states'):
                    self.verified_model.all_adapter_hidden_states = old_all_adapter_hidden_states

                if hasattr(self.verified_model, 'all_confidence_scores'):
                    self.verified_model.all_confidence_scores = old_all_confidence_scores

            # Calculate cosine similarity between true/false at each layer
            base_sims = []
            for t_state, f_state in zip(truth_base_states, falsehood_base_states):
                # Use the mean pooled representation
                t_pool = t_state.mean(dim=1).squeeze()
                f_pool = f_state.mean(dim=1).squeeze()

                # Calculate cosine similarity
                sim = F.cosine_similarity(t_pool, f_pool, dim=0).item()
                base_sims.append(sim)

            verified_sims = []
            for t_state, f_state in zip(truth_verified_states, falsehood_verified_states):
                # Use the mean pooled representation
                t_pool = t_state.mean(dim=1).squeeze()
                f_pool = f_state.mean(dim=1).squeeze()

                # Calculate cosine similarity
                sim = F.cosine_similarity(t_pool, f_pool, dim=0).item()
                verified_sims.append(sim)

            # Calculate verification impact (difference in confidence)
            verification_magnitude = []
            for t_conf, f_conf in zip(truth_confidence, falsehood_confidence):
                # Magnitude of the difference in confidence
                verification_magnitude.append(abs(t_conf - f_conf))

            # Find layers with biggest divergence
            base_divergence = [(i, 1-sim) for i, sim in enumerate(base_sims)]
            base_divergence.sort(key=lambda x: x[1], reverse=True)
            base_top_layers = base_divergence[:3]

            verified_divergence = [(i, 1-sim) for i, sim in enumerate(verified_sims)]
            verified_divergence.sort(key=lambda x: x[1], reverse=True)
            verified_top_layers = verified_divergence[:3]

            # Store results
            divergence_results["base_similarities"].append(base_sims)
            divergence_results["verified_similarities"].append(verified_sims)
            divergence_results["base_divergence_layers"].append(base_top_layers)
            divergence_results["verified_divergence_layers"].append(verified_top_layers)
            divergence_results["verification_magnitudes"].append(verification_magnitude)

        # Save results
        self.results["truth_falsehood_divergence"] = divergence_results

        # Create visualizations
        self._visualize_divergence(divergence_results)

        return divergence_results

    def _visualize_divergence(self, divergence_results):
        """Visualize divergence between true and false statements"""
        pairs = divergence_results["pairs"]
        base_sims = divergence_results["base_similarities"]
        verified_sims = divergence_results["verified_similarities"]
        verification_magnitudes = divergence_results["verification_magnitudes"]

        # 1. Plot similarity curves for each pair
        for i, ((truth, falsehood), base_sim, verified_sim) in enumerate(zip(pairs, base_sims, verified_sims)):
            plt.figure(figsize=(12, 6))

            # Plot similarities (higher = more similar)
            plt.plot(range(len(base_sim)), base_sim, 'b-', linewidth=2, label='Base Model')
            plt.plot(range(len(verified_sim)), verified_sim, 'r-', linewidth=2, label='Verified Model')

            # Get adapter locations if available
            adapter_locations = []
            if hasattr(self.verified_model, 'adapter_locations'):
                adapter_locations = self.verified_model.adapter_locations

                # Add markers for adapter locations
                for loc in adapter_locations:
                    if loc < len(verified_sim):
                        plt.axvline(x=loc, color='g', linestyle='--', alpha=0.5)

            plt.xlabel('Layer')
            plt.ylabel('Cosine Similarity')

            # Create abbreviated strings for title
            truth_abbr = truth[:40] + "..." if len(truth) > 40 else truth
            falsehood_abbr = falsehood[:40] + "..." if len(falsehood) > 40 else falsehood

            plt.title(f'True vs. False Representation Similarity\n"{truth_abbr}" vs. "{falsehood_abbr}"')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            # Add reference lines
            plt.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3)  # Perfect similarity
            plt.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3)  # No similarity

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"divergence_{i+1}.png"))
            plt.close()

        # 2. Plot verification magnitudes if available
        if verification_magnitudes and verification_magnitudes[0]:
            plt.figure(figsize=(12, 6))

            # Plot verification magnitude for each pair
            for i, (pair, magnitude) in enumerate(zip(pairs, verification_magnitudes)):
                truth, falsehood = pair
                # Create label
                if len(pairs) <= 5:  # Only include label if few pairs
                    label = f'"{truth[:20]}..." vs "{falsehood[:20]}..."'
                else:
                    label = f"Pair {i+1}"

                plt.plot(range(len(magnitude)), magnitude, 'o-', linewidth=2, label=label)

            plt.xlabel('Adapter Index')
            plt.ylabel('Confidence Difference Magnitude')
            plt.title('Verification Confidence Difference: True vs. False Statements')

            if len(pairs) <= 5:  # Only show legend if few pairs
                plt.legend()

            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "verification_magnitudes.png"))
            plt.close()

        # 3. Create a summary plot of average similarity by layer
        plt.figure(figsize=(12, 6))

        # Calculate average similarities
        base_avg = np.mean(base_sims, axis=0)
        verified_avg = np.mean(verified_sims, axis=0)

        # Plot average similarities
        plt.plot(range(len(base_avg)), base_avg, 'b-', linewidth=3, label='Base Model')
        plt.plot(range(len(verified_avg)), verified_avg, 'r-', linewidth=3, label='Verified Model')

        # Add shaded regions for standard deviation
        base_std = np.std(base_sims, axis=0)
        verified_std = np.std(verified_sims, axis=0)

        plt.fill_between(range(len(base_avg)), base_avg - base_std, base_avg + base_std, 
                        color='blue', alpha=0.2)
        plt.fill_between(range(len(verified_avg)), verified_avg - verified_std, verified_avg + verified_std, 
                        color='red', alpha=0.2)

        # Get adapter locations if available
        adapter_locations = []
        if hasattr(self.verified_model, 'adapter_locations'):
            adapter_locations = self.verified_model.adapter_locations

            # Add markers for adapter locations
            for loc in adapter_locations:
                if loc < len(verified_avg):
                    plt.axvline(x=loc, color='g', linestyle='--', alpha=0.5)

        plt.xlabel('Layer')
        plt.ylabel('Average Cosine Similarity')
        plt.title('Average True/False Representation Similarity by Layer')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add reference lines
        plt.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3)  # Perfect similarity
        plt.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3)  # No similarity

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "average_divergence.png"))
        plt.close()

        # 4. Create a heatmap of similarity differences
        if len(base_sims) >= 2:
            # Calculate differences
            sim_diffs = []
            for base, verified in zip(base_sims, verified_sims):
                # Only use up to minimum length
                min_len = min(len(base), len(verified))
                diff = np.array(verified[:min_len]) - np.array(base[:min_len])
                sim_diffs.append(diff)

            # Create heatmap
            plt.figure(figsize=(12, 8))
            diff_array = np.array(sim_diffs)

            sns.heatmap(diff_array, cmap="coolwarm", center=0,
                       xticklabels=[f"Layer {i}" for i in range(diff_array.shape[1])],
                       yticklabels=[f"Pair {i+1}" for i in range(diff_array.shape[0])])

            # Highlight adapter locations
            if adapter_locations:
                ax = plt.gca()
                for loc in adapter_locations:
                    if loc < diff_array.shape[1]:
                        # Add a rectangle around this column
                        rect = plt.Rectangle((loc, 0), 1, diff_array.shape[0], 
                                          linewidth=2, edgecolor='g', facecolor='none')
                        ax.add_patch(rect)

            plt.title('Similarity Difference: Verified - Base Model')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "similarity_difference_heatmap.png"))
            plt.close()

    def analyze_token_probability_flow(self, prompts=None):
        """
        Analyze how verification influences token prediction probabilities

        Args:
            prompts: List of prompts to analyze (if None, default prompts will be used)
        """
        if self.base_model is None or self.verified_model is None:
            self.load_models()

        logger.info("Analyzing token probability flow...")

        # Use default prompts if none provided
        if prompts is None:
            examples = self.prepare_test_examples()
            # Use reasoning examples that require completion
            prompts = examples["reasoning_examples"]

        # Results storage
        flow_results = {
            "prompts": prompts,
            "base_top_tokens": [],
            "verified_top_tokens": [],
            "base_probabilities": [],
            "verified_probabilities": [],
            "probability_shifts": []
        }

        # Process each prompt
        for prompt in prompts:
            logger.info(f"Processing prompt: {prompt}")

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Get predictions from base model
            with torch.no_grad():
                base_outputs = self.base_model(**inputs)
                base_logits = base_outputs.logits[:, -1, :]  # Logits for next token

                # Get top 10 tokens
                base_probs = F.softmax(base_logits, dim=-1)
                base_top_probs, base_top_indices = torch.topk(base_probs, 10)

                # Convert to lists
                base_top_indices = base_top_indices[0].tolist()
                base_top_probs = base_top_probs[0].tolist()

                # Decode tokens
                base_top_tokens = [self.tokenizer.decode([idx]) for idx in base_top_indices]

            # Get predictions from verified model
            with torch.no_grad():
                verified_outputs = self.verified_model(**inputs)
                verified_logits = verified_outputs.logits[:, -1, :]  # Logits for next token

                # Get top 10 tokens
                verified_probs = F.softmax(verified_logits, dim=-1)
                verified_top_probs, verified_top_indices = torch.topk(verified_probs, 10)

                # Convert to lists
                verified_top_indices = verified_top_indices[0].tolist()
                verified_top_probs = verified_top_probs[0].tolist()

                # Decode tokens
                verified_top_tokens = [self.tokenizer.decode([idx]) for idx in verified_top_indices]

            # Calculate probability shifts for interesting tokens
            prob_shifts = []

            # Combine top tokens from both models
            all_indices = set(base_top_indices + verified_top_indices)

            for idx in all_indices:
                base_prob = base_probs[0, idx].item()
                verified_prob = verified_probs[0, idx].item()

                # Calculate absolute and relative shift
                abs_shift = verified_prob - base_prob
                rel_shift = abs_shift / (base_prob + 1e-10)  # Avoid division by zero

                token = self.tokenizer.decode([idx])

                prob_shifts.append({
                    "token": token,
                    "token_id": idx,
                    "base_prob": base_prob,
                    "verified_prob": verified_prob,
                    "abs_shift": abs_shift,
                    "rel_shift": rel_shift
                })

            # Sort by absolute shift magnitude
            prob_shifts.sort(key=lambda x: abs(x["abs_shift"]), reverse=True)

            # Store results
            flow_results["base_top_tokens"].append(list(zip(base_top_tokens, base_top_probs)))
            flow_results["verified_top_tokens"].append(list(zip(verified_top_tokens, verified_top_probs)))
            flow_results["base_probabilities"].append((base_top_indices, base_top_probs))
            flow_results["verified_probabilities"].append((verified_top_indices, verified_top_probs))
            flow_results["probability_shifts"].append(prob_shifts)

        # Save results
        self.results["token_probability_flow"] = flow_results

        # Create visualizations
        self._visualize_probability_flow(flow_results)

        return flow_results

    def _visualize_probability_flow(self, flow_results):
        """Create visualizations of token probability flow"""
        prompts = flow_results["prompts"]
        base_top = flow_results["base_top_tokens"]
        verified_top = flow_results["verified_top_tokens"]
        prob_shifts = flow_results["probability_shifts"]

        # 1. Plot top token probabilities for each prompt
        for i, (prompt, base_tokens, verified_tokens) in enumerate(zip(prompts, base_top, verified_top)):
            plt.figure(figsize=(12, 8))

            # Get top 5 tokens from each model
            all_tokens = set([t for t, _ in base_tokens[:5]] + [t for t, _ in verified_tokens[:5]])
            tokens = sorted(list(all_tokens))

            # Get probabilities for these tokens
            base_probs = []
            verified_probs = []

            for token in tokens:
                # Find probability in base model
                base_prob = 0.0
                for t, p in base_tokens:
                    if t == token:
                        base_prob = p
                        break

                # Find probability in verified model
                verified_prob = 0.0
                for t, p in verified_tokens:
                    if t == token:
                        verified_prob = p
                        break

                base_probs.append(base_prob)
                verified_probs.append(verified_prob)

            # Prepare token labels (escape special characters)
            token_labels = []
            for t in tokens:
                # Escape underscores for matplotlib
                t = t.replace('_', '\_')
                if len(t) > 10:
                    t = t[:7] + '...'
                token_labels.append(t)

            # Create bar chart
            x = range(len(tokens))
            width = 0.35

            plt.bar([i - width/2 for i in x], base_probs, width, label='Base Model')
            plt.bar([i + width/2 for i in x], verified_probs, width, label='Verified Model')

            plt.xlabel('Tokens')
            plt.ylabel('Probability')

            # Create abbreviated prompt for title
            prompt_abbr = prompt[:50] + "..." if len(prompt) > 50 else prompt
            plt.title(f'Token Probabilities After Prompt:\n"{prompt_abbr}"')

            plt.xticks(x, token_labels, rotation=45, ha='right')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"token_probs_{i+1}.png"))
            plt.close()

        # 2. Plot top probability shifts for each prompt
        for i, (prompt, shifts) in enumerate(zip(prompts, prob_shifts)):
            plt.figure(figsize=(12, 8))

            # Take top 10 shifts by magnitude
            top_shifts = shifts[:10]

            # Get token labels and shifts
            token_labels = []
            abs_shifts = []
            base_probs = []
            verified_probs = []

            for shift in top_shifts:
                token = shift["token"].replace('_', '\_')  # Escape underscores
                if len(token) > 10:
                    token = token[:7] + '...'

                token_labels.append(token)
                abs_shifts.append(shift["abs_shift"])
                base_probs.append(shift["base_prob"])
                verified_probs.append(shift["verified_prob"])

            # Create a figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2]})

            # Plot 1: Absolute shifts
            colors = ['red' if s < 0 else 'green' for s in abs_shifts]
            ax1.bar(token_labels, abs_shifts, color=colors)
            ax1.set_ylabel('Probability Shift')
            ax1.set_title('Absolute Probability Shifts')
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            # Plot 2: Before/after comparison
            x = range(len(token_labels))
            width = 0.35

            ax2.bar([i - width/2 for i in x], base_probs, width, label='Base Model')
            ax2.bar([i + width/2 for i in x], verified_probs, width, label='Verified Model')

            ax2.set_xlabel('Tokens')
            ax2.set_ylabel('Probability')
            ax2.set_title('Token Probabilities Before/After Verification')
            ax2.set_xticks(x)
            ax2.set_xticklabels(token_labels, rotation=45, ha='right')
            ax2.legend()

            # Create abbreviated prompt for figure title
            prompt_abbr = prompt[:50] + "..." if len(prompt) > 50 else prompt
            fig.suptitle(f'Token Probability Flow Analysis\nPrompt: "{prompt_abbr}"', fontsize=12)

            plt.tight_layout()
            plt.subplots_adjust(top=0.9)  # Make room for suptitle
            plt.savefig(os.path.join(self.output_dir, f"probability_shifts_{i+1}.png"))
            plt.close()

    def create_latent_thinking_visualization(self, prompt=None):
        """
        Create a comprehensive visualization that illustrates the
        "thinking in latent space" concept for a single prompt

        Args:
            prompt: The prompt to visualize (if None, a default will be used)
        """
        if self.base_model is None or self.verified_model is None:
            self.load_models()

        # Use default prompt if none provided
        if prompt is None:
            prompt = "The human heart has three chambers."

        logger.info(f"Creating latent thinking visualization for: {prompt}")

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Set up comprehensive analysis
        analysis = {
            "prompt": prompt,
            "base_hidden_states": None,
            "verified_hidden_states": None,
            "adapter_locations": [],
            "verification_confidence": [],
            "layer_distances": [],
            "top_token_shifts": [],
            "attention_patterns": []
        }

        # Get adapter locations if available
        if hasattr(self.verified_model, 'adapter_locations'):
            analysis["adapter_locations"] = self.verified_model.adapter_locations

        # Process with base model
        with torch.no_grad():
            base_outputs = self.base_model(**inputs, output_hidden_states=True, output_attentions=True)

            # Store hidden states
            analysis["base_hidden_states"] = [h.detach().cpu() for h in base_outputs.hidden_states]

            # Get attention patterns (optional)
            if base_outputs.attentions:
                # Average attention over heads for visualization
                base_attentions = [attn.mean(dim=1).detach().cpu() for attn in base_outputs.attentions]
                analysis["base_attentions"] = base_attentions

        # Process with verified model
        with torch.no_grad():
            # Reset adapter states
            if hasattr(self.verified_model, 'all_adapter_hidden_states'):
                old_all_adapter_hidden_states = self.verified_model.all_adapter_hidden_states
                self.verified_model.all_adapter_hidden_states = []

            if hasattr(self.verified_model, 'all_confidence_scores'):
                old_all_confidence_scores = self.verified_model.all_confidence_scores
                self.verified_model.all_confidence_scores = []

            verified_outputs = self.verified_model(**inputs, output_hidden_states=True, output_attentions=True)

            # Store hidden states
            analysis["verified_hidden_states"] = [h.detach().cpu() for h in verified_outputs.hidden_states]

            # Get confidence scores
            if hasattr(self.verified_model, 'all_confidence_scores') and self.verified_model.all_confidence_scores:
                analysis["verification_confidence"] = [c.mean().item() for c in self.verified_model.all_confidence_scores]

            # Get attention patterns (optional)
            if verified_outputs.attentions:
                # Average attention over heads for visualization
                verified_attentions = [attn.mean(dim=1).detach().cpu() for attn in verified_outputs.attentions]
                analysis["verified_attentions"] = verified_attentions

            # Restore adapter states
            if hasattr(self.verified_model, 'all_adapter_hidden_states'):
                self.verified_model.all_adapter_hidden_states = old_all_adapter_hidden_states

            if hasattr(self.verified_model, 'all_confidence_scores'):
                self.verified_model.all_confidence_scores = old_all_confidence_scores

        # Calculate layer distances
        base_distances = []
        verified_distances = []

        for i in range(1, len(analysis["base_hidden_states"])):
            # Base model distances
            prev = analysis["base_hidden_states"][i-1]
            curr = analysis["base_hidden_states"][i]
            dist = (curr - prev).norm().item() / (prev.numel() ** 0.5)
            base_distances.append(dist)

            # Verified model distances
            prev = analysis["verified_hidden_states"][i-1]
            curr = analysis["verified_hidden_states"][i]
            dist = (curr - prev).norm().item() / (prev.numel() ** 0.5)
            verified_distances.append(dist)

        analysis["base_distances"] = base_distances
        analysis["verified_distances"] = verified_distances

        # Get token probability shifts
        analysis["token_shifts"] = self._calculate_token_shifts(prompt)

        # Create comprehensive visualization
        self._create_comprehensive_visualization(analysis)

        return analysis

    def _calculate_token_shifts(self, prompt):
        """Calculate token probability shifts for next token prediction"""
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Get predictions from base model
        with torch.no_grad():
            base_outputs = self.base_model(**inputs)
            base_logits = base_outputs.logits[:, -1, :]  # Logits for next token

            # Get probabilities
            base_probs = F.softmax(base_logits, dim=-1)
            base_top_probs, base_top_indices = torch.topk(base_probs, 20)

            # Convert to lists
            base_top_indices = base_top_indices[0].tolist()
            base_top_probs = base_top_probs[0].tolist()

        # Get predictions from verified model
        with torch.no_grad():
            verified_outputs = self.verified_model(**inputs)
            verified_logits = verified_outputs.logits[:, -1, :]  # Logits for next token

            # Get probabilities
            verified_probs = F.softmax(verified_logits, dim=-1)

        # Calculate shifts for top tokens
        shifts = []

        for idx, base_prob in zip(base_top_indices, base_top_probs):
            verified_prob = verified_probs[0, idx].item()

            # Calculate shift
            abs_shift = verified_prob - base_prob
            rel_shift = abs_shift / (base_prob + 1e-10)

            # Decode token
            token = self.tokenizer.decode([idx])

            shifts.append({
                "token": token,
                "token_id": idx,
                "base_prob": base_prob,
                "verified_prob": verified_prob,
                "abs_shift": abs_shift,
                "rel_shift": rel_shift
            })

        # Sort by absolute shift magnitude
        shifts.sort(key=lambda x: abs(x["abs_shift"]), reverse=True)

        return shifts

    def _create_comprehensive_visualization(self, analysis):
        """Create a comprehensive visualization of latent thinking"""
        prompt = analysis["prompt"]
        base_distances = analysis["base_distances"]
        verified_distances = analysis["verified_distances"]
        adapter_locations = analysis["adapter_locations"]
        verification_confidence = analysis.get("verification_confidence", [])
        token_shifts = analysis.get("token_shifts", [])

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 20))
        gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1])

        # 1. Trajectory plot (top left)
        ax1 = fig.add_subplot(gs[0, 0])

        # Plot distances
        ax1.plot(range(len(base_distances)), base_distances, 'b-', linewidth=2, label='Base Model')
        ax1.plot(range(len(verified_distances)), verified_distances, 'r-', linewidth=2, label='Verified Model')

        # Add markers for adapter locations
        for loc in adapter_locations:
            if loc < len(verified_distances):
                ax1.axvline(x=loc, color='g', linestyle='--', alpha=0.5)

        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Hidden State Change')
        ax1.set_title('Hidden State Trajectory')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

        # 2. Verification confidence plot (top right)
        ax2 = fig.add_subplot(gs[0, 1])

        if verification_confidence:
            # Plot confidence scores
            x = range(len(verification_confidence))
            ax2.bar(x, verification_confidence, color='purple', alpha=0.7)

            ax2.set_xlabel('Adapter Index')
            ax2.set_ylabel('Confidence Score')
            ax2.set_title('Verification Confidence')
            ax2.set_ylim(0, 1)
            ax2.grid(True, linestyle='--', alpha=0.7)
        else:
            ax2.text(0.5, 0.5, "Confidence scores not available", 
                    ha='center', va='center', fontsize=12)
            ax2.set_title('Verification Confidence')
            ax2.axis('off')

        # 3. Layer trajectory PCA visualization (middle left)
        ax3 = fig.add_subplot(gs[1, 0])

        # Use PCA on hidden states
        try:
            # Get first token representations
            base_tokens = [states[:, 0, :].squeeze().numpy() for states in analysis["base_hidden_states"]]
            verified_tokens = [states[:, 0, :].squeeze().numpy() for states in analysis["verified_hidden_states"]]

            # Combine all states for PCA
            all_states = np.vstack(base_tokens + verified_tokens)

            # Apply PCA
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(all_states)

            # Split back into base and verified
            n_layers = len(base_tokens)
            base_reduced = reduced[:n_layers]
            verified_reduced = reduced[n_layers:]

            # Plot trajectories
            ax3.plot(base_reduced[:, 0], base_reduced[:, 1], 'b-', linewidth=2, alpha=0.7, label='Base Model')
            ax3.plot(verified_reduced[:, 0], verified_reduced[:, 1], 'r-', linewidth=2, alpha=0.7, label='Verified Model')

            # Add points for each layer
            for i, (x, y) in enumerate(base_reduced):
                ax3.scatter(x, y, color='blue', s=100, alpha=0.7)
                ax3.text(x, y, str(i), color='white', ha='center', va='center', fontsize=8)

            for i, (x, y) in enumerate(verified_reduced):
                if i in adapter_locations:
                    ax3.scatter(x, y, color='red', s=150, alpha=0.7, marker='*')
                else:
                    ax3.scatter(x, y, color='red', s=100, alpha=0.7)
                ax3.text(x, y, str(i), color='white', ha='center', va='center', fontsize=8)

            ax3.set_title('Hidden State Trajectory (PCA)')
            ax3.set_xlabel('PC1')
            ax3.set_ylabel('PC2')
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.7)

        except Exception as e:
            logger.warning(f"Error in PCA visualization: {e}")
            ax3.text(0.5, 0.5, "PCA visualization failed", ha='center', va='center', fontsize=12)
            ax3.set_title('Hidden State Trajectory (PCA)')
            ax3.axis('off')

        # 4. Token probability shifts (middle right)
        ax4 = fig.add_subplot(gs[1, 1])

        if token_shifts:
            # Take top 10 shifts
            top_shifts = token_shifts[:10]

            # Extract data
            tokens = [s["token"] for s in top_shifts]
            shifts = [s["abs_shift"] for s in top_shifts]

            # Create bar chart with color based on shift direction
            colors = ['red' if s < 0 else 'green' for s in shifts]
            bars = ax4.bar(tokens, shifts, color=colors)

            ax4.set_xlabel('Tokens')
            ax4.set_ylabel('Probability Shift')
            ax4.set_title('Token Probability Shifts')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.set_xticklabels(tokens, rotation=45, ha='right')

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height >= 0:
                    va = 'bottom'
                    offset = 0.01
                else:
                    va = 'top'
                    offset = -0.01
                ax4.text(bar.get_x() + bar.get_width()/2., height + offset,
                        f'{height:.3f}', ha='center', va=va, fontsize=8)

        else:
            ax4.text(0.5, 0.5, "Token shifts not available", 
                    ha='center', va='center', fontsize=12)
            ax4.set_title('Token Probability Shifts')
            ax4.axis('off')

        # 5. Difference trajectory visualization (bottom left)
        ax5 = fig.add_subplot(gs[2, 0])

        # Calculate differences between verified and base distances
        if len(base_distances) == len(verified_distances):
            differences = np.array(verified_distances) - np.array(base_distances)

            # Plot differences
            bars = ax5.bar(range(len(differences)), differences, 
                         color=['red' if d < 0 else 'green' for d in differences])

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height >= 0:
                    va = 'bottom'
                    offset = 0.01 * max(abs(min(differences)), max(differences))
                else:
                    va = 'top'
                    offset = -0.01 * max(abs(min(differences)), max(differences))
                ax5.text(bar.get_x() + bar.get_width()/2., height + offset,
                        f'{height:.3f}', ha='center', va=va, fontsize=8)

            ax5.set_xlabel('Layer')
            ax5.set_ylabel('Distance Difference')
            ax5.set_title('Hidden State Change Difference (Verified - Base)')
            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax5.grid(True, axis='y', linestyle='--', alpha=0.7)

            # Highlight adapter locations
            for loc in adapter_locations:
                if loc < len(differences):
                    ax5.axvline(x=loc, color='g', linestyle='--', alpha=0.5)
        else:
            ax5.text(0.5, 0.5, "Model distance arrays have different lengths", 
                    ha='center', va='center', fontsize=12)
            ax5.set_title('Hidden State Change Difference')
            ax5.axis('off')

        # 6. Before/After token probabilities (bottom right)
        ax6 = fig.add_subplot(gs[2, 1])

        if token_shifts:
            # Take top 5 shifts
            top_shifts = token_shifts[:5]

            # Extract data
            tokens = [s["token"] for s in top_shifts]
            base_probs = [s["base_prob"] for s in top_shifts]
            verified_probs = [s["verified_prob"] for s in top_shifts]

            # Create bar chart
            x = range(len(tokens))
            width = 0.35

            ax6.bar([i - width/2 for i in x], base_probs, width, label='Base Model')
            ax6.bar([i + width/2 for i in x], verified_probs, width, label='Verified Model')

            ax6.set_xlabel('Tokens')
            ax6.set_ylabel('Probability')
            ax6.set_title('Token Probabilities Before/After Verification')
            ax6.set_xticks(x)
            ax6.set_xticklabels(tokens, rotation=45, ha='right')
            ax6.legend()

        else:
            ax6.text(0.5, 0.5, "Token shifts not available", 
                    ha='center', va='center', fontsize=12)
            ax6.set_title('Token Probabilities')
            ax6.axis('off')

        # 7. Conceptual visualization of latent thinking (bottom)
        ax7 = fig.add_subplot(gs[3, :])

        # Create conceptual visualization
        self._create_conceptual_visualization(ax7, analysis)

        # Figure title
        fig.suptitle(f'Latent Thinking Analysis: "{prompt}"', fontsize=16, y=0.99)

        plt.tight_layout()
        plt.subplots_adjust(top=0.96)  # Make room for suptitle

        # Save figure
        plt.savefig(os.path.join(self.output_dir, "latent_thinking_comprehensive.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_conceptual_visualization(self, ax, analysis):
        """Create a conceptual visualization of latent thinking process"""
        # Clear axis
        ax.clear()
        ax.axis('off')

        # Define the stages of the latent thinking process
        stages = [
            "Input", "Hidden\nRepresentation", "Verification\nAdapter", 
            "Correction", "Refined\nRepresentation", "Output"
        ]

        # Define x positions for each stage
        x_positions = [0.1, 0.25, 0.4, 0.55, 0.7, 0.9]

        # Get confirmation confidence if available
        confidence = None
        if analysis.get("verification_confidence"):
            confidence = np.mean(analysis["verification_confidence"])

        # Get token shifts if available
        token_shifts = analysis.get("token_shifts", [])
        top_token = None
        if token_shifts:
            # Find token with largest positive shift
            positive_shifts = [s for s in token_shifts if s["abs_shift"] > 0]
            if positive_shifts:
                top_token = positive_shifts[0]["token"]

        # Draw stage bubbles and arrows
        bubbles = []
        for i, (stage, x) in enumerate(zip(stages, x_positions)):
            # Create bubble
            if i == 2:  # Verification adapter
                # Use star shape for verification
                color = 'purple'
                bubble = ax.scatter(x, 0.5, s=1500, marker='*', color=color, alpha=0.6)
            else:
                color = 'blue' if i in [0, 1, 5] else 'red'
                bubble = Circle((x, 0.5), 0.1, color=color, alpha=0.6)
                ax.add_patch(bubble)

            # Add label
            ax.text(x, 0.5, stage, ha='center', va='center', fontweight='bold', color='white')

            # Add extra annotations
            if i == 2 and confidence is not None:  # Verification adapter
                ax.text(x, 0.3, f"Confidence: {confidence:.3f}", ha='center', va='center', fontsize=10)

            if i == 3:  # Correction
                ax.text(x, 0.3, "Latent Space Thinking", ha='center', va='center', fontsize=10)

            if i == 5 and top_token:  # Output
                ax.text(x, 0.3, f"Preferred: {top_token}", ha='center', va='center', fontsize=10)

            bubbles.append(bubble)

        # Add arrows between stages
        for i in range(len(stages) - 1):
            # Create arrow
            start_x = x_positions[i] + 0.1
            end_x = x_positions[i+1] - 0.1

            arrow = ax.annotate("", xy=(end_x, 0.5), xytext=(start_x, 0.5),
                             arrowprops=dict(arrowstyle="->", lw=2, color='black'))

            # Add label for verification arrow
            if i == 1:
                ax.text((start_x + end_x) / 2, 0.6, "Verification", ha='center', va='center', fontsize=10)

            # Add label for correction arrow
            if i == 2:
                ax.text((start_x + end_x) / 2, 0.6, "Apply Correction", ha='center', va='center', fontsize=10)

        # Set limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Add title
        ax.set_title("Conceptual Model of Latent Space Thinking and Verification", fontsize=14)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze latent thinking in verification models")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name")
    parser.add_argument("--verified_model", type=str, required=True, help="Verified model path")
    parser.add_argument("--output_dir", type=str, default="latent_thinking_analysis", help="Output directory")
    parser.add_argument("--trajectories", action="store_true", help="Run hidden state trajectory analysis")
    parser.add_argument("--divergence", action="store_true", help="Run truth/falsehood divergence analysis")
    parser.add_argument("--token_flow", action="store_true", help="Run token probability flow analysis")
    parser.add_argument("--comprehensive", action="store_true", help="Create comprehensive visualization")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    parser.add_argument("--prompt", type=str, help="Custom prompt for comprehensive visualization")

    args = parser.parse_args()

    # Determine which analyses to run
    run_all = args.all
    run_trajectories = args.trajectories or run_all
    run_divergence = args.divergence or run_all
    run_token_flow = args.token_flow or run_all
    run_comprehensive = args.comprehensive or run_all

    # If no specific analyses are selected, run comprehensive by default
    if not (run_trajectories or run_divergence or run_token_flow or run_comprehensive):
        run_comprehensive = True

    # Initialize analyzer
    analyzer = LatentThinkingAnalyzer(
        base_model_name=args.base_model,
        verified_model_path=args.verified_model,
        output_dir=args.output_dir
    )

    # Load models
    analyzer.load_models()

    # Run selected analyses
    if run_trajectories:
        analyzer.analyze_hidden_state_trajectories()

    if run_divergence:
        analyzer.analyze_truth_falsehood_divergence()

    if run_token_flow:
        analyzer.analyze_token_probability_flow()

    if run_comprehensive:
        analyzer.create_latent_thinking_visualization(args.prompt)

