"""
Confidence Analysis Tool for Latent Verification

This tool provides a detailed analysis of how verification confidence scores
correlate with factual correctness. It helps validate the core hypothesis that
confidence scores from verification adapters meaningfully reflect the factual
accuracy of the content being processed.

Key features:
1. Confidence-accuracy correlation analysis
2. Confidence distribution visualization for true vs. false content
3. Layer-by-layer confidence analysis
4. Confidence calibration evaluation (reliability diagrams)
5. Statistical significance testing of confidence differences
"""

import os, sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats
import argparse
from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("confidence_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class ConfidenceAnalyzer:
    """Analyze verification confidence scores and their relationship to factual accuracy"""

    def __init__(
        self,
        verified_model_path: str,
        base_model_name: Optional[str] = None,
        output_dir: str = "confidence_analysis",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the confidence analyzer

        Args:
            verified_model_path: Path to the verification-enhanced model
            base_model_name: Name of the base model (optional, for comparison)
            output_dir: Directory to save analysis outputs
            device: Device to run model on (cuda or cpu)
        """
        self.verified_model_path = verified_model_path
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.device = device

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Will be initialized when needed
        self.tokenizer = None
        self.verified_model = None
        self.base_model = None

        # Results storage
        self.confidence_data = []
        self.layer_confidences = {}
        self.calibration_data = {}

        logger.info(f"Initializing confidence analyzer for model at {verified_model_path}")
        logger.info(f"Using device: {device}")

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

    def analyze_truth_falsehood_confidence(self, truth_falsehood_pairs: List[Tuple[str, str]] = None):
        """
        Analyze confidence scores for true vs. false statements

        Args:
            truth_falsehood_pairs: List of (true_statement, false_statement) pairs
                If None, default examples will be used
        """
        if self.verified_model is None:
            self.load_models()

        # Use default pairs if not provided
        if truth_falsehood_pairs is None:
            truth_falsehood_pairs = [
                # Factual knowledge
                ("The capital of France is Paris.", "The capital of France is Berlin."),
                ("The Earth orbits around the Sun.", "The Sun orbits around the Earth."),
                ("Water boils at 100 degrees Celsius at sea level.", "Water boils at 50 degrees Celsius at sea level."),
                ("The human heart has four chambers.", "The human heart has three chambers."),
                ("Mount Everest is the tallest mountain in the world.", "K2 is the tallest mountain in the world."),
                ("The Pacific Ocean is the largest ocean on Earth.", "The Atlantic Ocean is the largest ocean on Earth."),
                ("Oxygen has the chemical symbol O.", "Oxygen has the chemical symbol Ox."),
                ("The Great Wall of China is visible from space.", "The Great Wall of China is not visible from space."),
                ("Spiders are arachnids, not insects.", "Spiders are insects, not arachnids."),
                ("The United States declared independence in 1776.", "The United States declared independence in 1789."),

                # Mathematical facts
                ("Two plus two equals four.", "Two plus two equals five."),
                ("The square root of 16 is 4.", "The square root of 16 is 5."),
                ("A right angle is 90 degrees.", "A right angle is 100 degrees."),
                ("A circle has 360 degrees.", "A circle has 380 degrees."),
                ("A triangle has three sides.", "A triangle has four sides."),

                # Scientific facts
                ("DNA has a double helix structure.", "DNA has a triple helix structure."),
                ("The Earth's atmosphere is mostly nitrogen.", "The Earth's atmosphere is mostly oxygen."),
                ("Electrons have a negative charge.", "Electrons have a positive charge."),
                ("The speed of light is faster than the speed of sound.", "The speed of sound is faster than the speed of light."),
                ("Photosynthesis is the process by which plants make food.", "Photosynthesis is the process by which animals digest food.")
            ]

        logger.info(f"Analyzing confidence scores for {len(truth_falsehood_pairs)} truth/falsehood pairs...")

        # Reset data storage
        self.confidence_data = []
        self.layer_confidences = {
            "truth": {},
            "falsehood": {}
        }

        # Process each pair
        for i, (truth, falsehood) in enumerate(tqdm(truth_falsehood_pairs, desc="Processing pairs")):
            # Analyze true statement
            truth_confidences, truth_avg = self._get_confidence_scores(truth)

            # Analyze false statement
            falsehood_confidences, falsehood_avg = self._get_confidence_scores(falsehood)

            # Store data
            pair_data = {
                "pair_id": i,
                "truth": truth,
                "falsehood": falsehood,
                "truth_confidence_avg": truth_avg,
                "falsehood_confidence_avg": falsehood_avg,
                "confidence_difference": truth_avg - falsehood_avg
            }

            self.confidence_data.append(pair_data)

            # Store layer-specific confidences
            for layer_idx, conf in truth_confidences.items():
                if layer_idx not in self.layer_confidences["truth"]:
                    self.layer_confidences["truth"][layer_idx] = []
                self.layer_confidences["truth"][layer_idx].append(conf)

            for layer_idx, conf in falsehood_confidences.items():
                if layer_idx not in self.layer_confidences["falsehood"]:
                    self.layer_confidences["falsehood"][layer_idx] = []
                self.layer_confidences["falsehood"][layer_idx].append(conf)

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.confidence_data)

        # Calculate average confidence for true and false statements
        avg_truth_confidence = df["truth_confidence_avg"].mean()
        avg_falsehood_confidence = df["falsehood_confidence_avg"].mean()
        avg_confidence_diff = df["confidence_difference"].mean()

        logger.info(f"Average confidence for true statements: {avg_truth_confidence:.4f}")
        logger.info(f"Average confidence for false statements: {avg_falsehood_confidence:.4f}")
        logger.info(f"Average confidence difference: {avg_confidence_diff:.4f}")

        # Statistical significance test
        t_stat, p_value = stats.ttest_rel(df["truth_confidence_avg"], df["falsehood_confidence_avg"])
        logger.info(f"Paired t-test: t={t_stat:.4f}, p={p_value:.6f}")

        # Generate visualizations
        self._visualize_confidence_distribution(df)
        self._visualize_layer_confidence(self.layer_confidences)
        self._visualize_confidence_ranking(df)

        # Save analysis results
        results = {
            "avg_truth_confidence": avg_truth_confidence,
            "avg_falsehood_confidence": avg_falsehood_confidence,
            "avg_confidence_diff": avg_confidence_diff,
            "t_statistic": t_stat,
            "p_value": p_value,
            "statistically_significant": p_value < 0.05,
            "num_samples": len(df),
            "confidence_data": df.to_dict(orient="records")
        }

        with open(os.path.join(self.output_dir, "confidence_analysis.json"), "w") as f:
            json.dump(convert_numpy_types(results), f, indent=2)

        return results

    def _get_confidence_scores(self, text: str) -> Tuple[Dict[int, float], float]:
        """
        Get confidence scores for a text input

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (layer_confidences, average_confidence)
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

        # Forward pass
        with torch.no_grad():
            outputs = self.verified_model(**inputs, output_hidden_states=True)

        # Get confidence scores
        layer_confidences = {}

        # Try to get confidence scores from model hooks
        if hasattr(self.verified_model, 'all_confidence_scores') and self.verified_model.all_confidence_scores:
            adapter_locations = self.verified_model.adapter_locations if hasattr(self.verified_model, 'adapter_locations') else range(len(self.verified_model.all_confidence_scores))

            for i, (layer_idx, conf) in enumerate(zip(adapter_locations, self.verified_model.all_confidence_scores)):
                layer_confidences[layer_idx] = conf.mean().item()

        # Try to get confidence from verification_metrics if hooks didn't work
        elif hasattr(outputs, 'verification_metrics') and "layer_confidence_scores" in outputs.verification_metrics:
            adapter_locations = self.verified_model.adapter_locations if hasattr(self.verified_model, 'adapter_locations') else range(len(outputs.verification_metrics["layer_confidence_scores"]))

            for i, (layer_idx, conf) in enumerate(zip(adapter_locations, outputs.verification_metrics["layer_confidence_scores"])):
                layer_confidences[layer_idx] = conf.mean().item()

        # Calculate average confidence
        if layer_confidences:
            avg_confidence = sum(layer_confidences.values()) / len(layer_confidences)
        else:
            # If no confidence scores found, use a default
            avg_confidence = 0.5
            logger.warning("No confidence scores found. Using default value 0.5.")

        # Restore adapter states if they were saved
        if hasattr(self.verified_model, 'all_adapter_hidden_states') and old_all_adapter_hidden_states is not None:
            self.verified_model.all_adapter_hidden_states = old_all_adapter_hidden_states

        if hasattr(self.verified_model, 'all_confidence_scores') and old_all_confidence_scores is not None:
            self.verified_model.all_confidence_scores = old_all_confidence_scores

        return layer_confidences, avg_confidence

    def _visualize_confidence_distribution(self, df: pd.DataFrame):
        """
        Visualize the distribution of confidence scores for true vs. false statements

        Args:
            df: DataFrame containing confidence data
        """
        plt.figure(figsize=(10, 6))

        # Plot confidence distributions
        sns.histplot(df["truth_confidence_avg"], label="True Statements", alpha=0.6, color="green", bins=20)
        sns.histplot(df["falsehood_confidence_avg"], label="False Statements", alpha=0.6, color="red", bins=20)

        # Add vertical lines for means
        plt.axvline(x=df["truth_confidence_avg"].mean(), color="green", linestyle="--", alpha=0.8)
        plt.axvline(x=df["falsehood_confidence_avg"].mean(), color="red", linestyle="--", alpha=0.8)

        plt.xlabel("Confidence Score")
        plt.ylabel("Frequency")
        plt.title("Distribution of Confidence Scores for True vs. False Statements")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.savefig(os.path.join(self.output_dir, "confidence_distribution.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Plot confidence difference distribution
        plt.figure(figsize=(10, 6))

        sns.histplot(df["confidence_difference"], color="blue", bins=20)
        plt.axvline(x=df["confidence_difference"].mean(), color="black", linestyle="--", alpha=0.8)
        plt.axvline(x=0, color="red", linestyle="-", alpha=0.5)

        plt.xlabel("Confidence Difference (True - False)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Confidence Differences")
        plt.grid(alpha=0.3)

        plt.savefig(os.path.join(self.output_dir, "confidence_difference.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Plot violin plot for comparison
        plt.figure(figsize=(8, 6))

        plot_data = pd.DataFrame({
            "Confidence": df["truth_confidence_avg"].tolist() + df["falsehood_confidence_avg"].tolist(),
            "Type": ["True"] * len(df) + ["False"] * len(df)
        })

        sns.violinplot(x="Type", y="Confidence", hue="Type", data=plot_data, palette={"True": "green", "False": "red"}, legend=False)
        sns.stripplot(x="Type", y="Confidence", hue="Type", data=plot_data, alpha=0.6, jitter=True, size=4, palette={"True": "darkgreen", "False": "darkred"}, legend=False)

        plt.ylim(0, 1)
        plt.title("Confidence Score Comparison: True vs. False Statements")
        plt.grid(axis='y', alpha=0.3)

        plt.savefig(os.path.join(self.output_dir, "confidence_violin.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def _visualize_layer_confidence(self, layer_confidences: Dict[str, Dict[int, List[float]]]):
        """
        Visualize confidence scores across layers for true vs. false statements

        Args:
            layer_confidences: Dictionary of layer-specific confidences
        """
        # Ensure we have layer data
        if not layer_confidences["truth"] or not layer_confidences["falsehood"]:
            logger.warning("No layer-specific confidence data available.")
            return

        # Get common layers
        layers = sorted(set(layer_confidences["truth"].keys()) & set(layer_confidences["falsehood"].keys()))

        if not layers:
            logger.warning("No common layers found in confidence data.")
            return

        # Calculate average confidence per layer
        truth_means = [np.mean(layer_confidences["truth"][layer]) for layer in layers]
        falsehood_means = [np.mean(layer_confidences["falsehood"][layer]) for layer in layers]

        # Calculate confidence differences per layer
        differences = [truth - falsehood for truth, falsehood in zip(truth_means, falsehood_means)]

        # Plot confidence by layer
        plt.figure(figsize=(12, 6))

        plt.plot(layers, truth_means, 'g-o', label="True Statements", linewidth=2, markersize=8)
        plt.plot(layers, falsehood_means, 'r-o', label="False Statements", linewidth=2, markersize=8)

        plt.xlabel("Layer")
        plt.ylabel("Average Confidence Score")
        plt.title("Confidence Scores Across Layers: True vs. False Statements")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.savefig(os.path.join(self.output_dir, "layer_confidence.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Plot confidence differences by layer
        plt.figure(figsize=(12, 6))

        plt.bar(layers, differences, color='blue', alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        plt.xlabel("Layer")
        plt.ylabel("Confidence Difference (True - False)")
        plt.title("Confidence Differences Across Layers")
        plt.grid(axis='y', alpha=0.3)

        plt.savefig(os.path.join(self.output_dir, "layer_confidence_difference.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Create heatmap with confidence by layer and statement
        plt.figure(figsize=(14, 10))

        # Prepare data for heatmap
        truth_layer_data = {}
        falsehood_layer_data = {}

        for layer in layers:
            truth_values = layer_confidences["truth"][layer]
            falsehood_values = layer_confidences["falsehood"][layer]

            # Ensure equal length for heatmap
            min_len = min(len(truth_values), len(falsehood_values))

            for i in range(min_len):
                pair_id = i

                if pair_id not in truth_layer_data:
                    truth_layer_data[pair_id] = {}
                    falsehood_layer_data[pair_id] = {}

                truth_layer_data[pair_id][layer] = truth_values[i]
                falsehood_layer_data[pair_id][layer] = falsehood_values[i]

        # Convert to DataFrames
        truth_df = pd.DataFrame(truth_layer_data).T
        falsehood_df = pd.DataFrame(falsehood_layer_data).T

        # Create heatmaps
        plt.subplot(2, 1, 1)
        sns.heatmap(truth_df, cmap="Greens", annot=True, fmt=".2f", linewidths=0.5)
        plt.title("Confidence Scores for True Statements by Layer")
        plt.xlabel("Layer")
        plt.ylabel("Statement Pair ID")

        plt.subplot(2, 1, 2)
        sns.heatmap(falsehood_df, cmap="Reds", annot=True, fmt=".2f", linewidths=0.5)
        plt.title("Confidence Scores for False Statements by Layer")
        plt.xlabel("Layer")
        plt.ylabel("Statement Pair ID")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "confidence_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def _visualize_confidence_ranking(self, df: pd.DataFrame):
        """
        Visualize confidence scores ranked by difference

        Args:
            df: DataFrame containing confidence data
        """
        # Sort by confidence difference
        df_sorted = df.sort_values(by="confidence_difference", ascending=False).reset_index(drop=True)

        # Plot top statements with highest confidence difference
        plt.figure(figsize=(12, 8))

        # Get top and bottom 10 pairs (or fewer if there are fewer pairs)
        num_pairs = min(10, len(df_sorted) // 2)
        top_pairs = df_sorted.head(num_pairs)
        bottom_pairs = df_sorted.tail(num_pairs)

        # Combine and sort for visualization
        plot_pairs = pd.concat([top_pairs, bottom_pairs])
        plot_pairs = plot_pairs.sort_values(by="confidence_difference", ascending=True).reset_index(drop=True)

        # Create bar plot
        plt.barh(plot_pairs.index, plot_pairs["confidence_difference"],
                color=[("green" if x > 0 else "red") for x in plot_pairs["confidence_difference"]])

        # Add statement texts as labels
        for i, row in plot_pairs.iterrows():
            # Format text to show both statements
            if row["confidence_difference"] > 0:
                # Positive difference (true > false)
                text_pos = 0.01
                diff_text = f"+{row['confidence_difference']:.3f}"
            else:
                # Negative difference (false > true)
                text_pos = row["confidence_difference"] + 0.01
                diff_text = f"{row['confidence_difference']:.3f}"

            # Add confidence difference
            plt.text(text_pos, i, diff_text, va='center', fontsize=9)

            # Truncate statements if too long
            truth_text = row["truth"][:50] + "..." if len(row["truth"]) > 50 else row["truth"]
            falsehood_text = row["falsehood"][:50] + "..." if len(row["falsehood"]) > 50 else row["falsehood"]

            # Add statements as y-labels
            plt.text(-0.01, i, f"{i+1}. T: {truth_text} | F: {falsehood_text}", ha='right', va='center', fontsize=8)

        plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        plt.xlim(-0.6, 0.6)
        plt.xlabel("Confidence Difference (True - False)")
        plt.title("Statements with Highest and Lowest Confidence Differences")
        plt.yticks([])  # Hide y-ticks since we have custom labels
        plt.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "confidence_ranking.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def analyze_confidence_calibration(self, calibration_data: List[Dict[str, Any]] = None):
        """
        Analyze how well confidence scores are calibrated with actual correctness

        Args:
            calibration_data: List of dictionaries with keys:
                - "input": input text
                - "is_correct": boolean indicating if statement is factually correct
                If None, true/false pairs from previous analysis will be used
        """
        if self.verified_model is None:
            self.load_models()

        # Use data from previous analysis if not provided
        if calibration_data is None:
            if not self.confidence_data:
                logger.warning("No confidence data available. Run analyze_truth_falsehood_confidence first or provide calibration_data.")
                return None

            # Create calibration data from previous analysis
            calibration_data = []
            for pair in self.confidence_data:
                # Add true statement
                calibration_data.append({
                    "input": pair["truth"],
                    "is_correct": True
                })

                # Add false statement
                calibration_data.append({
                    "input": pair["falsehood"],
                    "is_correct": False
                })

        logger.info(f"Analyzing confidence calibration for {len(calibration_data)} statements...")

        # Get confidence scores for all inputs
        confidences = []
        correctness = []

        for item in tqdm(calibration_data, desc="Calculating confidence scores"):
            _, confidence = self._get_confidence_scores(item["input"])
            confidences.append(confidence)
            correctness.append(1 if item["is_correct"] else 0)

        # Store calibration data
        self.calibration_data = {
            "confidences": confidences,
            "correctness": correctness
        }

        # Generate calibration curve
        prob_true, prob_pred = calibration_curve(correctness, confidences, n_bins=10)

        # Calculate calibration metrics
        ece = np.mean(np.abs(prob_true - prob_pred))  # Expected Calibration Error

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(correctness, confidences)
        roc_auc = auc(fpr, tpr)

        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(correctness, confidences)
        pr_auc = average_precision_score(correctness, confidences)

        logger.info(f"Expected Calibration Error (ECE): {ece:.4f}")
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        logger.info(f"PR AUC: {pr_auc:.4f}")

        # Visualize calibration
        self._visualize_calibration(prob_true, prob_pred, fpr, tpr, roc_auc, precision, recall, pr_auc)

        # Save calibration results
        results = {
            "expected_calibration_error": ece,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "calibration_curve": {
                "prob_true": prob_true.tolist(),
                "prob_pred": prob_pred.tolist()
            },
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist()
            },
            "pr_curve": {
                "precision": precision.tolist(),
                "recall": recall.tolist()
            }
        }

        with open(os.path.join(self.output_dir, "calibration_analysis.json"), "w") as f:
            json.dump(results, f, indent=2)

        return results

    def _visualize_calibration(self, prob_true, prob_pred, fpr, tpr, roc_auc, precision, recall, pr_auc):
        """
        Visualize confidence calibration

        Args:
            prob_true: True probabilities from calibration curve
            prob_pred: Predicted probabilities from calibration curve
            fpr: False positive rates for ROC curve
            tpr: True positive rates for ROC curve
            roc_auc: Area under ROC curve
            precision: Precision values for PR curve
            recall: Recall values for PR curve
            pr_auc: Area under PR curve
        """
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Calibration curve
        axs[0, 0].plot(prob_pred, prob_true, marker='o', linewidth=2)
        axs[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.7)  # Diagonal perfect calibration line
        axs[0, 0].set_xlabel('Mean Predicted Confidence')
        axs[0, 0].set_ylabel('Fraction of Correct Statements')
        axs[0, 0].set_title('Confidence Calibration Curve')
        axs[0, 0].grid(alpha=0.3)

        # 2. Confidence histogram by correctness
        confidences = np.array(self.calibration_data["confidences"])
        correctness = np.array(self.calibration_data["correctness"])

        axs[0, 1].hist(confidences[correctness == 1], alpha=0.7, bins=20, color='green', label='Correct')
        axs[0, 1].hist(confidences[correctness == 0], alpha=0.7, bins=20, color='red', label='Incorrect')
        axs[0, 1].set_xlabel('Confidence Score')
        axs[0, 1].set_ylabel('Frequency')
        axs[0, 1].set_title('Confidence Distribution by Correctness')
        axs[0, 1].legend()
        axs[0, 1].grid(alpha=0.3)

        # 3. ROC curve
        axs[1, 0].plot(fpr, tpr, linewidth=2)
        axs[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.7)  # Diagonal random classifier line
        axs[1, 0].set_xlabel('False Positive Rate')
        axs[1, 0].set_ylabel('True Positive Rate')
        axs[1, 0].set_title(f'ROC Curve (AUC = {roc_auc:.3f})')
        axs[1, 0].grid(alpha=0.3)

        # 4. Precision-Recall curve
        axs[1, 1].plot(recall, precision, linewidth=2)
        axs[1, 1].set_xlabel('Recall')
        axs[1, 1].set_ylabel('Precision')
        axs[1, 1].set_title(f'Precision-Recall Curve (AUC = {pr_auc:.3f})')
        axs[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "calibration_analysis.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def analyze_confidence_over_time(self, texts: List[str] = None, generate_length: int = 20):
        """
        Analyze how confidence evolves during text generation

        Args:
            texts: List of prompt texts to analyze confidence over generation
            generate_length: Number of tokens to generate for each text
        """
        if self.verified_model is None:
            self.load_models()

        # Use default texts if not provided
        if texts is None:
            texts = [
                "The capital of France is",
                "The tallest mountain in the world is",
                "The number of planets in our solar system is",
                "The chemical symbol for gold is",
                "The speed of light is approximately"
            ]

        logger.info(f"Analyzing confidence over generation time for {len(texts)} prompts...")

        # Results storage
        generation_results = []

        for prompt_idx, prompt in enumerate(texts):
            logger.info(f"Processing prompt {prompt_idx+1}: '{prompt}'")

            # Generate text token by token to track confidence
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            generated_text = self.tokenizer.decode(input_ids[0])

            # Track confidence, tokens, and hidden states
            confidence_history = []
            token_history = []

            # Generate tokens one by one
            for _ in range(generate_length):
                # Generate next token
                with torch.no_grad():
                    # Reset adapter states
                    if hasattr(self.verified_model, 'all_adapter_hidden_states'):
                        old_all_adapter_hidden_states = self.verified_model.all_adapter_hidden_states
                        self.verified_model.all_adapter_hidden_states = []

                    if hasattr(self.verified_model, 'all_confidence_scores'):
                        old_all_confidence_scores = self.verified_model.all_confidence_scores
                        self.verified_model.all_confidence_scores = []

                    # Get model outputs
                    outputs = self.verified_model(input_ids, output_hidden_states=True)

                    # Get next token logits
                    next_token_logits = outputs.logits[:, -1, :]

                    # Get next token
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

                    # Get token text
                    next_token_text = self.tokenizer.decode(next_token_id[0])
                    token_history.append(next_token_text)

                    # Get confidence scores
                    layer_confidences = {}

                    # Try to get confidence scores from model hooks
                    if hasattr(self.verified_model, 'all_confidence_scores') and self.verified_model.all_confidence_scores:
                        adapter_locations = self.verified_model.adapter_locations if hasattr(self.verified_model, 'adapter_locations') else range(len(self.verified_model.all_confidence_scores))

                        for i, (layer_idx, conf) in enumerate(zip(adapter_locations, self.verified_model.all_confidence_scores)):
                            layer_confidences[layer_idx] = conf.mean().item()

                    # Try to get confidence from verification_metrics if hooks didn't work
                    elif hasattr(outputs, 'verification_metrics') and "layer_confidence_scores" in outputs.verification_metrics:
                        adapter_locations = self.verified_model.adapter_locations if hasattr(self.verified_model, 'adapter_locations') else range(len(outputs.verification_metrics["layer_confidence_scores"]))

                        for i, (layer_idx, conf) in enumerate(zip(adapter_locations, outputs.verification_metrics["layer_confidence_scores"])):
                            layer_confidences[layer_idx] = conf.mean().item()

                    # Calculate average confidence
                    if layer_confidences:
                        avg_confidence = sum(layer_confidences.values()) / len(layer_confidences)
                    else:
                        # If no confidence scores found, use a default
                        avg_confidence = 0.5

                    # Store confidence
                    confidence_history.append({
                        "token_idx": len(token_history) - 1,
                        "token": next_token_text,
                        "avg_confidence": avg_confidence,
                        "layer_confidences": layer_confidences
                    })

                    # Restore adapter states
                    if hasattr(self.verified_model, 'all_adapter_hidden_states'):
                        self.verified_model.all_adapter_hidden_states = old_all_adapter_hidden_states

                    if hasattr(self.verified_model, 'all_confidence_scores'):
                        self.verified_model.all_confidence_scores = old_all_confidence_scores

                # Append to input_ids for next iteration
                input_ids = torch.cat([input_ids, next_token_id], dim=1)

                # Update generated text
                generated_text = self.tokenizer.decode(input_ids[0])

            # Store results for this prompt
            generation_results.append({
                "prompt": prompt,
                "generated_text": generated_text,
                "token_history": token_history,
                "confidence_history": confidence_history
            })

        # Visualize confidence over generation
        self._visualize_generation_confidence(generation_results)

        # Save generation results
        with open(os.path.join(self.output_dir, "generation_confidence.json"), "w") as f:
            json.dump(generation_results, f, indent=2)

        return generation_results

    def _visualize_generation_confidence(self, generation_results: List[Dict[str, Any]]):
        """
        Visualize confidence over text generation

        Args:
            generation_results: List of generation result dictionaries
        """
        # One plot per prompt
        for idx, result in enumerate(generation_results):
            plt.figure(figsize=(14, 6))

            # Extract data
            tokens = [item["token"] for item in result["confidence_history"]]
            confidences = [item["avg_confidence"] for item in result["confidence_history"]]

            # Plot confidence over tokens
            plt.plot(range(len(confidences)), confidences, 'o-', linewidth=2, markersize=8)

            # Add token labels
            for i, (token, conf) in enumerate(zip(tokens, confidences)):
                # Clean token for display
                clean_token = token.replace('\n', '\\n')
                if len(clean_token) > 10:
                    clean_token = clean_token[:10] + "..."

                plt.text(i, conf + 0.02, clean_token, ha='center', fontsize=9, rotation=45)

            # Add prompt
            plt.title(f"Confidence Evolution During Generation\nPrompt: \"{result['prompt']}\"")
            plt.xlabel("Token Position")
            plt.ylabel("Confidence Score")
            plt.ylim(0, 1)
            plt.grid(alpha=0.3)

            # Add complete generated text as a text box
            text_box = f"Complete generated text:\n{result['generated_text']}"
            plt.figtext(0.5, 0.01, text_box, ha="center", fontsize=10, 
                      bbox={"facecolor":"white", "alpha":0.5, "pad":5})

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)  # Make room for the text box
            plt.savefig(os.path.join(self.output_dir, f"generation_confidence_{idx+1}.png"), dpi=300, bbox_inches="tight")
            plt.close()

        # Combined plot for all prompts
        plt.figure(figsize=(12, 8))

        for result in generation_results:
            confidences = [item["avg_confidence"] for item in result["confidence_history"]]
            prompt_text = result["prompt"][:20] + "..." if len(result["prompt"]) > 20 else result["prompt"]
            plt.plot(range(len(confidences)), confidences, 'o-', linewidth=2, markersize=4, label=prompt_text)

        plt.title("Confidence Evolution During Generation Across Prompts")
        plt.xlabel("Token Position")
        plt.ylabel("Confidence Score")
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "combined_generation_confidence.png"), dpi=300, bbox_inches="tight")
        plt.close()

def main():
    """Run the confidence analysis tool from command line"""
    parser = argparse.ArgumentParser(description="Analyze verification confidence scores")
    parser.add_argument("--verified_model", type=str, required=True, help="Path to verification-enhanced model")
    parser.add_argument("--base_model", type=str, default=None, help="Base model name (optional)")
    parser.add_argument("--output_dir", type=str, default="confidence_analysis", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--run_all", action="store_true", help="Run all analyses")
    parser.add_argument("--analyze_confidence", action="store_true", help="Run confidence analysis")
    parser.add_argument("--analyze_calibration", action="store_true", help="Run calibration analysis")
    parser.add_argument("--analyze_generation", action="store_true", help="Run generation confidence analysis")

    args = parser.parse_args()

    # Create analyzer
    analyzer = ConfidenceAnalyzer(
        verified_model_path=args.verified_model,
        base_model_name=args.base_model,
        output_dir=args.output_dir,
        device=args.device
    )

    # Determine which analyses to run
    run_confidence = args.run_all or args.analyze_confidence
    run_calibration = args.run_all or args.analyze_calibration
    run_generation = args.run_all or args.analyze_generation

    # Run selected analyses
    if run_confidence:
        analyzer.analyze_truth_falsehood_confidence()

    if run_calibration:
        analyzer.analyze_confidence_calibration()

    if run_generation:
        analyzer.analyze_confidence_over_time()

if __name__ == "__main__":
    main()
