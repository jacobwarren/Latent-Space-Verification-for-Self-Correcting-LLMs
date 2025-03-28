import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random
import time
from collections import defaultdict
import torch.nn.functional as F

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VerificationEvaluator:
    """
    Comprehensive evaluation suite for latent-space verification mechanisms
    """
    def __init__(
        self,
        base_model_name: str,
        verified_model_path: str,
        output_dir: str = "evaluation_results",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the evaluator with model paths

        Args:
            base_model_name: Name or path of the original base model
            verified_model_path: Path to the model with verification components
            output_dir: Directory to save evaluation results
            device: Device to run evaluations on
        """
        self.base_model_name = base_model_name
        self.verified_model_path = verified_model_path
        self.output_dir = output_dir
        self.device = device

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Evaluation results storage
        self.results = {}

        logger.info(f"Initializing evaluator with base model: {base_model_name}")
        logger.info(f"Verified model path: {verified_model_path}")
        logger.info(f"Device: {device}")

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Ensure proper padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load models (will be loaded when specific evaluations are run)
        self.base_model = None
        self.verified_model = None

        # For parameter-matched baselines
        self.adapter_only_model = None
        self.lora_model = None
        self.mlp_adapter_model = None

        # Tracking metrics and test cases
        self.test_cases = {}
        self.metrics = defaultdict(dict)

    def load_models(self):
        """Load all required models for evaluation"""
        # Load base model
        logger.info("Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name
        ).to(self.device)

        # Load verified model
        logger.info("Loading verified model...")

        try:
            # First, try loading with the LatentVerificationWrapper
            from enhanced_verification import load_bayesian_verification_model
            self.verified_model = load_bayesian_verification_model(
                self.verified_model_path
            ).to(self.device)
            logger.info("Successfully loaded verified model with wrapper")
        except Exception as e:
            logger.warning(f"Error loading with wrapper: {e}. Trying standard loading...")
            # Fallback to standard loading
            self.verified_model = AutoModelForCausalLM.from_pretrained(
                self.verified_model_path
            ).to(self.device)
            logger.info("Loaded verified model with standard loading")

    def load_parameter_matched_baselines(self, adapter_model_path=None, lora_model_path=None, mlp_model_path=None):
        """Load parameter-matched baseline models for comparison"""
        if adapter_model_path:
            logger.info(f"Loading adapter-only model from {adapter_model_path}")
            self.adapter_only_model = AutoModelForCausalLM.from_pretrained(adapter_model_path).to(self.device)

        if lora_model_path:
            logger.info(f"Loading LoRA model from {lora_model_path}")
            self.lora_model = AutoModelForCausalLM.from_pretrained(lora_model_path).to(self.device)

        if mlp_model_path:
            logger.info(f"Loading MLP adapter model from {mlp_model_path}")
            self.mlp_adapter_model = AutoModelForCausalLM.from_pretrained(mlp_model_path).to(self.device)

    def compare_parameter_counts(self):
        """Compare parameter counts across models"""
        if self.base_model is None or self.verified_model is None:
            self.load_models()

        logger.info("Comparing parameter counts across models...")

        # Count parameters
        base_params = sum(p.numel() for p in self.base_model.parameters())
        verified_params = sum(p.numel() for p in self.verified_model.parameters())

        # Count trainable parameters
        verified_trainable = sum(p.numel() for p in self.verified_model.parameters() if p.requires_grad)

        # Parameter efficiency ratio
        efficiency_ratio = verified_trainable / base_params

        # Store results
        params_info = {
            "base_total": base_params,
            "verified_total": verified_params,
            "verified_trainable": verified_trainable,
            "parameter_increase": verified_params - base_params,
            "parameter_increase_percent": (verified_params - base_params) / base_params * 100,
            "efficiency_ratio": efficiency_ratio,
        }

        # Include parameter-matched baselines if available
        if self.adapter_only_model:
            adapter_params = sum(p.numel() for p in self.adapter_only_model.parameters())
            adapter_trainable = sum(p.numel() for p in self.adapter_only_model.parameters() if p.requires_grad)
            params_info["adapter_only_total"] = adapter_params
            params_info["adapter_only_trainable"] = adapter_trainable

        if self.lora_model:
            lora_params = sum(p.numel() for p in self.lora_model.parameters())
            lora_trainable = sum(p.numel() for p in self.lora_model.parameters() if p.requires_grad)
            params_info["lora_total"] = lora_params
            params_info["lora_trainable"] = lora_trainable

        if self.mlp_adapter_model:
            mlp_params = sum(p.numel() for p in self.mlp_adapter_model.parameters())
            mlp_trainable = sum(p.numel() for p in self.mlp_adapter_model.parameters() if p.requires_grad)
            params_info["mlp_adapter_total"] = mlp_params
            params_info["mlp_adapter_trainable"] = mlp_trainable

        # Save results
        self.results["parameter_counts"] = params_info

        # Log parameter counts
        logger.info(f"Base model parameters: {base_params:,}")
        logger.info(f"Verified model parameters: {verified_params:,}")
        logger.info(f"Verified model trainable parameters: {verified_trainable:,}")
        logger.info(f"Parameter increase: {params_info['parameter_increase']:,} ({params_info['parameter_increase_percent']:.2f}%)")
        logger.info(f"Parameter efficiency ratio: {efficiency_ratio:.6f}")

        # Plot parameter comparison
        self._plot_parameter_comparison(params_info)

        return params_info

    def _plot_parameter_comparison(self, params_info):
        """Create bar chart comparing parameter counts"""
        plt.figure(figsize=(12, 8))

        # Prepare data for plotting
        models = ["Base Model"]
        total_params = [params_info["base_total"] / 1_000_000]  # Convert to millions
        trainable_params = [0]  # Base model has no trainable params in fine-tuning

        models.append("Verified Model")
        total_params.append(params_info["verified_total"] / 1_000_000)
        trainable_params.append(params_info["verified_trainable"] / 1_000_000)

        # Add other models if available
        if "adapter_only_total" in params_info:
            models.append("Adapter Only")
            total_params.append(params_info["adapter_only_total"] / 1_000_000)
            trainable_params.append(params_info["adapter_only_trainable"] / 1_000_000)

        if "lora_total" in params_info:
            models.append("LoRA")
            total_params.append(params_info["lora_total"] / 1_000_000)
            trainable_params.append(params_info["lora_trainable"] / 1_000_000)

        if "mlp_adapter_total" in params_info:
            models.append("MLP Adapter")
            total_params.append(params_info["mlp_adapter_total"] / 1_000_000)
            trainable_params.append(params_info["mlp_adapter_trainable"] / 1_000_000)

        # Set up bar positions
        x = np.arange(len(models))
        width = 0.35

        # Create grouped bars
        plt.bar(x - width/2, total_params, width, label='Total Parameters')
        plt.bar(x + width/2, trainable_params, width, label='Trainable Parameters')

        plt.ylabel('Parameters (millions)')
        plt.title('Parameter Comparison Across Models')
        plt.xticks(x, models, rotation=45)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "parameter_comparison.png"))
        plt.close()

        # Create pie chart for verified model
        plt.figure(figsize=(10, 6))
        verified_frozen = params_info["verified_total"] - params_info["verified_trainable"]
        sizes = [verified_frozen, params_info["verified_trainable"]]
        labels = ['Frozen Parameters', 'Trainable Parameters']
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Verified Model Parameter Breakdown')
        plt.savefig(os.path.join(self.output_dir, "verified_model_parameter_pie.png"))
        plt.close()

    def run_factual_knowledge_evaluation(self, dataset_name="truthful_qa", dataset_config="multiple_choice", sample_size=100):
        """
        Evaluate models on factual knowledge using benchmarks like TruthfulQA

        Args:
            dataset_name: Name of the dataset to use
            dataset_config: Configuration name for the dataset
            sample_size: Number of samples to evaluate (use None for full dataset)
        """
        if self.base_model is None or self.verified_model is None:
            self.load_models()

        logger.info(f"Running factual knowledge evaluation using {dataset_name}/{dataset_config}...")

        # Load the dataset
        try:
            dataset = load_dataset(dataset_name, dataset_config)
            eval_split = "validation" if "validation" in dataset else "test"
            eval_dataset = dataset[eval_split]

            # Debug: Print the full structure of the first example
            if len(eval_dataset) > 0:
                logger.info(f"First example structure: {eval_dataset[0]}")
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}/{dataset_config}: {e}")
            # Fallback to minimal TruthfulQA-like test set
            eval_dataset = self._create_minimal_factual_test()
            logger.info(f"Using minimal factual test set with {len(eval_dataset)} examples")

        # Track results
        results = {
            "base_model": {"correct": 0, "total": 0, "accuracy": 0},
            "verified_model": {"correct": 0, "total": 0, "accuracy": 0},
        }

        # Also track parameter-matched baselines if available
        if self.adapter_only_model:
            results["adapter_only_model"] = {"correct": 0, "total": 0, "accuracy": 0}
        if self.lora_model:
            results["lora_model"] = {"correct": 0, "total": 0, "accuracy": 0}
        if self.mlp_adapter_model:
            results["mlp_adapter_model"] = {"correct": 0, "total": 0, "accuracy": 0}

        # Prepare for detailed results storage
        detailed_results = []

        # Process each example
        for i, example in enumerate(tqdm(eval_dataset, desc="Evaluating factual knowledge")):
            question, choices, correct_idx = self._extract_qa_fields(example, dataset_name)

            if not question or choices is None or correct_idx is None:
                logger.warning(f"Skipping example {i}: could not extract required fields")
                continue

            # Format the input for the model
            input_text = f"Question: {question}\n\nChoices:\n"
            for j, choice in enumerate(choices):
                input_text += f"{j+1}. {choice}\n"
            input_text += "\nProvide the number of the correct answer:"

            # Process with base model
            base_output = self._get_model_answer(self.base_model, input_text)
            base_pred = self._extract_answer_idx(base_output, len(choices))
            base_correct = (base_pred == correct_idx)
            results["base_model"]["correct"] += int(base_correct)
            results["base_model"]["total"] += 1

            # Process with verified model
            verified_output = self._get_model_answer(self.verified_model, input_text)
            verified_pred = self._extract_answer_idx(verified_output, len(choices))
            verified_correct = (verified_pred == correct_idx)
            results["verified_model"]["correct"] += int(verified_correct)
            results["verified_model"]["total"] += 1

            # Process with parameter-matched baselines if available
            if self.adapter_only_model:
                adapter_output = self._get_model_answer(self.adapter_only_model, input_text)
                adapter_pred = self._extract_answer_idx(adapter_output, len(choices))
                adapter_correct = (adapter_pred == correct_idx)
                results["adapter_only_model"]["correct"] += int(adapter_correct)
                results["adapter_only_model"]["total"] += 1
            else:
                adapter_pred = None
                adapter_correct = None

            if self.lora_model:
                lora_output = self._get_model_answer(self.lora_model, input_text)
                lora_pred = self._extract_answer_idx(lora_output, len(choices))
                lora_correct = (lora_pred == correct_idx)
                results["lora_model"]["correct"] += int(lora_correct)
                results["lora_model"]["total"] += 1
            else:
                lora_pred = None
                lora_correct = None

            if self.mlp_adapter_model:
                mlp_output = self._get_model_answer(self.mlp_adapter_model, input_text)
                mlp_pred = self._extract_answer_idx(mlp_output, len(choices))
                mlp_correct = (mlp_pred == correct_idx)
                results["mlp_adapter_model"]["correct"] += int(mlp_correct)
                results["mlp_adapter_model"]["total"] += 1
            else:
                mlp_pred = None
                mlp_correct = None

            # Store detailed results for this example
            example_result = {
                "question": question,
                "choices": choices,
                "correct_idx": correct_idx,
                "base_pred": base_pred,
                "base_correct": base_correct,
                "verified_pred": verified_pred,
                "verified_correct": verified_correct,
                "base_output": base_output,
                "verified_output": verified_output,
            }

            # Add results from parameter-matched baselines if available
            if adapter_pred is not None:
                example_result["adapter_pred"] = adapter_pred
                example_result["adapter_correct"] = adapter_correct

            if lora_pred is not None:
                example_result["lora_pred"] = lora_pred
                example_result["lora_correct"] = lora_correct

            if mlp_pred is not None:
                example_result["mlp_pred"] = mlp_pred
                example_result["mlp_correct"] = mlp_correct

            detailed_results.append(example_result)

        # Calculate final accuracies
        for model_name in results:
            if results[model_name]["total"] > 0:
                results[model_name]["accuracy"] = results[model_name]["correct"] / results[model_name]["total"]

        # Log results
        logger.info(f"Base model accuracy: {results['base_model']['accuracy']:.4f}")
        logger.info(f"Verified model accuracy: {results['verified_model']['accuracy']:.4f}")

        if self.adapter_only_model:
            logger.info(f"Adapter-only model accuracy: {results['adapter_only_model']['accuracy']:.4f}")
        if self.lora_model:
            logger.info(f"LoRA model accuracy: {results['lora_model']['accuracy']:.4f}")
        if self.mlp_adapter_model:
            logger.info(f"MLP adapter model accuracy: {results['mlp_adapter_model']['accuracy']:.4f}")

        # Store results
        self.results["factual_knowledge"] = {
            "dataset": dataset_name,
            "config": dataset_config,
            "sample_size": len(detailed_results),
            "summary": results,
            "detailed": detailed_results
        }

        # Plot results
        self._plot_factual_knowledge_results(results)

        # Analyze error patterns
        self._analyze_error_patterns(detailed_results)

        return results

    def _extract_qa_fields(self, example, dataset_name):
        """Extract question, choices, and correct answer from dataset example"""
        if dataset_name == "truthful_qa":
            # TruthfulQA multiple choice format
            question = example.get("question", "")

            # Handle different formats
            if "mc1_targets" in example:
                choices = example["mc1_targets"].get("choices", [])

                # Check if mc1_labels exists
                if "mc1_labels" in example:
                    correct_idx = example["mc1_labels"]["labels"].index(1) if 1 in example["mc1_labels"]["labels"] else 0
                else:
                    # When mc1_labels is missing, the first choice is usually the correct one in mc1_targets
                    correct_idx = 0

            elif "choices" in example and "label" in example:
                choices = example["choices"]
                correct_idx = example["label"]
            else:
                logger.warning(f"Unexpected TruthfulQA format: {example.keys()}")
                return None, None, None

        else:
            # Generic format - try to extract from common field names
            question = example.get("question", example.get("query", example.get("prompt", "")))

            # Try to find choices and correct answer
            choices = example.get("choices", example.get("options", example.get("answers", [])))
            correct_idx = example.get("correct_idx", example.get("label", example.get("answer_idx", 0)))

            # Handle case where correct answer is provided as text
            correct_answer = example.get("correct_answer", example.get("answer", ""))
            if correct_answer and not isinstance(correct_idx, int) and isinstance(choices, list):
                # Try to find the correct answer in choices
                try:
                    correct_idx = choices.index(correct_answer)
                except ValueError:
                    correct_idx = 0  # Fallback

        return question, choices, correct_idx

    def _get_model_answer(self, model, input_text, max_new_tokens=50):
        """Get model's answer to a question"""
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=False
            )

        # Decode the output
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the answer part
        answer = full_output[len(input_text):].strip()
        return answer

    def _extract_answer_idx(self, answer_text, num_choices):
        """Extract the answer index from model output"""
        # First look for a single digit response
        for char in answer_text:
            if char.isdigit() and 1 <= int(char) <= num_choices:
                return int(char) - 1  # Convert to 0-indexed

        # Look for "Answer: X" or similar patterns
        patterns = [
            r"Answer: (\d+)",
            r"answer is (\d+)",
            r"option (\d+)",
            r"choice (\d+)"
        ]

        import re
        for pattern in patterns:
            match = re.search(pattern, answer_text, re.IGNORECASE)
            if match and 1 <= int(match.group(1)) <= num_choices:
                return int(match.group(1)) - 1

        # Default to first option if no valid answer found
        return 0

    def _create_minimal_factual_test(self):
        """Create a minimal test set for factual evaluation when dataset loading fails"""
        test_set = [
            {
                "question": "Which planet is known as the Red Planet?",
                "choices": ["Venus", "Mars", "Jupiter", "Saturn"],
                "correct_idx": 1
            },
            {
                "question": "Who wrote 'Romeo and Juliet'?",
                "choices": ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"],
                "correct_idx": 1
            },
            {
                "question": "What is the capital of Japan?",
                "choices": ["Beijing", "Seoul", "Tokyo", "Bangkok"],
                "correct_idx": 2
            },
            # Add more examples as needed
        ]
        return test_set

    def _plot_factual_knowledge_results(self, results):
        """Plot the results of factual knowledge evaluation"""
        plt.figure(figsize=(10, 6))

        # Prepare data for plotting
        models = []
        accuracies = []

        for model_name, result in results.items():
            if result["total"] > 0:  # Only include models with results
                models.append(model_name.replace("_", " ").title())
                accuracies.append(result["accuracy"] * 100)  # Convert to percentage

        # Create bar chart
        bars = plt.bar(models, accuracies)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')

        plt.ylabel('Accuracy (%)')
        plt.title('Factual Knowledge Accuracy Comparison')
        plt.ylim(0, max(accuracies) * 1.2)  # Add some space above bars

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "factual_knowledge_accuracy.png"))
        plt.close()

    def _analyze_error_patterns(self, detailed_results):
        """Analyze patterns in the errors made by each model"""
        # Count cases where models differ
        differ_count = sum(1 for r in detailed_results if r["base_correct"] != r["verified_correct"])
        base_only_correct = sum(1 for r in detailed_results if r["base_correct"] and not r["verified_correct"])
        verified_only_correct = sum(1 for r in detailed_results if not r["base_correct"] and r["verified_correct"])

        total = len(detailed_results)

        # Log analysis
        logger.info(f"Error pattern analysis:")
        logger.info(f"  Total examples: {total}")
        logger.info(f"  Models disagree on {differ_count} examples ({differ_count/total*100:.1f}%)")
        logger.info(f"  Base model correct, verified model wrong: {base_only_correct} ({base_only_correct/total*100:.1f}%)")
        logger.info(f"  Verified model correct, base model wrong: {verified_only_correct} ({verified_only_correct/total*100:.1f}%)")

        # Create confusion matrix-like visualization
        plt.figure(figsize=(8, 8))
        matrix = np.zeros((2, 2))

        # Fill the matrix
        both_correct = sum(1 for r in detailed_results if r["base_correct"] and r["verified_correct"])
        both_wrong = sum(1 for r in detailed_results if not r["base_correct"] and not r["verified_correct"])

        matrix[0, 0] = both_wrong
        matrix[0, 1] = verified_only_correct
        matrix[1, 0] = base_only_correct
        matrix[1, 1] = both_correct

        # Plot as heatmap
        sns.heatmap(matrix, annot=True, fmt='.0f', cmap='Blues',
           xticklabels=["Verified Wrong", "Verified Correct"],
           yticklabels=["Base Wrong", "Base Correct"])
        plt.title("Error Pattern Comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "error_pattern_matrix.png"))
        plt.close()

        # Store the analysis
        self.results["error_patterns"] = {
            "total_examples": total,
            "models_disagree": differ_count,
            "base_only_correct": base_only_correct,
            "verified_only_correct": verified_only_correct,
            "both_correct": both_correct,
            "both_wrong": both_wrong
        }

    def analyze_verification_activations(self, test_prompts=None, num_samples=5):
        """
        Analyze the verification mechanism's activations on test inputs

        Args:
            test_prompts: List of prompts to test (if None, default prompts will be used)
            num_samples: Number of samples to use if test_prompts is None
        """
        if self.verified_model is None:
            self.load_models()

        logger.info("Analyzing verification mechanism activations...")

        # Default test prompts if none provided
        if test_prompts is None:
            test_prompts = [
                "The capital of France is Berlin.",
                "The tallest mountain in the world is Mount Everest.",
                "The human heart has three chambers.",
                "The sun revolves around the Earth.",
                "Water boils at 100 degrees Celsius at sea level.",
                "The Earth is flat.",
                "Gravity causes objects to fall upward.",
                "Humans share 50% of their DNA with bananas.",
                "The Great Wall of China is visible from space.",
                "Vaccines cause autism."
            ]
            # Limit to requested number of samples
            test_prompts = test_prompts[:num_samples]

        # Check if the verified model has verification components
        has_verification_adapters = hasattr(self.verified_model, 'verification_adapters')
        has_cross_layer_verifier = hasattr(self.verified_model, 'cross_layer_verifier')

        if not has_verification_adapters:
            logger.warning("Verified model does not have verification_adapters attribute. Limited analysis possible.")

        results = []

        for prompt in test_prompts:
            logger.info(f"Processing prompt: {prompt}")

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Forward pass through verified model
            with torch.no_grad():
                # Store original hooks
                old_all_adapter_hidden_states = getattr(self.verified_model, 'all_adapter_hidden_states', [])
                old_all_confidence_scores = getattr(self.verified_model, 'all_confidence_scores', [])

                # Reset collections
                self.verified_model.all_adapter_hidden_states = []
                self.verified_model.all_confidence_scores = []

                # Run forward pass
                outputs = self.verified_model(**inputs, output_hidden_states=True)

                # Collect verification metrics
                verification_metrics = getattr(outputs, 'verification_metrics', {})

                # Check if we have access to confidence scores
                confidence_scores = []
                if hasattr(self.verified_model, 'all_confidence_scores') and self.verified_model.all_confidence_scores:
                    confidence_scores = [score.mean().item() for score in self.verified_model.all_confidence_scores]
                elif "layer_confidence_scores" in verification_metrics:
                    confidence_scores = [score.mean().item() for score in verification_metrics["layer_confidence_scores"]]

                # Get cross-layer consistency if available
                cross_layer_consistency = None
                if "cross_layer_consistency" in verification_metrics:
                    cross_layer_consistency = verification_metrics["cross_layer_consistency"].mean().item()

                # Analyze hidden state modifications if available
                hidden_state_changes = []
                if hasattr(self.verified_model, 'all_adapter_hidden_states') and len(self.verified_model.all_adapter_hidden_states) > 0:
                    # Get original hidden states from the output
                    if hasattr(outputs, 'hidden_states'):
                        orig_hidden_states = outputs.hidden_states

                        # Match adapter locations to hidden states
                        adapter_locations = getattr(self.verified_model, 'adapter_locations', [])

                        for i, (adapter_idx, corrected_state) in enumerate(zip(adapter_locations, self.verified_model.all_adapter_hidden_states)):
                            if adapter_idx < len(orig_hidden_states):
                                orig_state = orig_hidden_states[adapter_idx]
                                # Calculate L2 norm of the difference
                                diff = (corrected_state - orig_state).norm().item()
                                # Normalize by tensor size
                                normalized_diff = diff / (orig_state.numel() ** 0.5)
                                hidden_state_changes.append(normalized_diff)

                # Store results for this prompt
                prompt_result = {
                    "prompt": prompt,
                    "confidence_scores": confidence_scores,
                    "cross_layer_consistency": cross_layer_consistency,
                    "hidden_state_changes": hidden_state_changes,
                    "logits": outputs.logits.mean().item(),
                }

                results.append(prompt_result)

                # Restore original hooks
                self.verified_model.all_adapter_hidden_states = old_all_adapter_hidden_states
                self.verified_model.all_confidence_scores = old_all_confidence_scores

        # Plot confidence scores across layers
        self._plot_confidence_scores(results)

        # Plot hidden state changes
        if any(len(r["hidden_state_changes"]) > 0 for r in results):
            self._plot_hidden_state_changes(results)

        # Store the results
        self.results["verification_activations"] = results

        return results

    def _plot_confidence_scores(self, results):
        """Plot confidence scores across layers for different prompts"""
        # Check if we have confidence scores
        if not results or not results[0]["confidence_scores"]:
            logger.warning("No confidence scores available for plotting")
            return

        plt.figure(figsize=(12, 8))

        # Get number of layers
        num_layers = len(results[0]["confidence_scores"])

        # Plot for each prompt
        for i, result in enumerate(results):
            confidence_scores = result["confidence_scores"]
            if confidence_scores:
                plt.plot(range(len(confidence_scores)), confidence_scores, 
                         marker='o', label=f"Prompt {i+1}", linewidth=2)

        plt.xlabel('Layer Index')
        plt.ylabel('Confidence Score')
        plt.title('Verification Confidence Scores Across Layers')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(range(num_layers))

        # Add prompt texts in the legend if there are few enough
        if len(results) <= 5:
            plt.legend([r["prompt"][:30] + "..." if len(r["prompt"]) > 30 else r["prompt"] for r in results])
        else:
            plt.legend([f"Prompt {i+1}" for i in range(len(results))])

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "confidence_scores.png"))
        plt.close()

        # Create heatmap if we have enough prompts
        if len(results) >= 3:
            plt.figure(figsize=(12, 8))

            # Prepare data for heatmap
            heatmap_data = np.array([r["confidence_scores"] for r in results])

            # Create heatmap
            sns.heatmap(heatmap_data, cmap="viridis", annot=True, fmt=".2f",
                        xticklabels=[f"Layer {i}" for i in range(num_layers)],
                        yticklabels=[f"Prompt {i+1}" for i in range(len(results))])

            plt.title('Confidence Score Heatmap')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "confidence_heatmap.png"))
            plt.close()

    def _plot_hidden_state_changes(self, results):
        """Plot hidden state changes across layers for different prompts"""
        plt.figure(figsize=(12, 8))

        # Get number of layers
        num_layers = max(len(r["hidden_state_changes"]) for r in results)

        # Plot for each prompt
        for i, result in enumerate(results):
            hidden_state_changes = result["hidden_state_changes"]
            if hidden_state_changes:
                plt.plot(range(len(hidden_state_changes)), hidden_state_changes, 
                         marker='o', label=f"Prompt {i+1}", linewidth=2)

        plt.xlabel('Layer Index')
        plt.ylabel('Normalized Hidden State Change')
        plt.title('Verification-Induced Hidden State Modifications')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(range(num_layers))

        # Add prompt texts in the legend if there are few enough
        if len(results) <= 5:
            plt.legend([r["prompt"][:30] + "..." if len(r["prompt"]) > 30 else r["prompt"] for r in results])
        else:
            plt.legend([f"Prompt {i+1}" for i in range(len(results))])

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "hidden_state_changes.png"))
        plt.close()

    def run_representational_analysis(self, test_prompts=None, num_samples=5):
        """
        Analyze and compare representations in the base and verified models

        Args:
            test_prompts: List of prompts to test (if None, default prompts will be used)
            num_samples: Number of samples to use if test_prompts is None
        """
        if self.base_model is None or self.verified_model is None:
            self.load_models()

        logger.info("Running representational analysis...")

        # Default test prompts if none provided
        if test_prompts is None:
            test_prompts = [
                "The capital of France is Paris.",
                "The capital of France is Berlin.",
                "The tallest mountain in the world is Mount Everest.",
                "The tallest mountain in the world is K2.",
                "The human heart has four chambers.",
                "The human heart has three chambers.",
                "The sun is at the center of our solar system.",
                "The Earth is at the center of our solar system.",
                "Water boils at 100 degrees Celsius at sea level.",
                "Water boils at 50 degrees Celsius at sea level."
            ]
            # Limit to requested number of pairs
            pairs = min(num_samples, len(test_prompts) // 2)
            test_prompts = test_prompts[:pairs*2]

        # Tokenize all prompts
        inputs = []
        for prompt in test_prompts:
            inputs.append(self.tokenizer(prompt, return_tensors="pt").to(self.device))

        # Get hidden states from both models
        base_hidden_states = []
        verified_hidden_states = []

        for i, input_ids in enumerate(inputs):
            # Process with base model
            with torch.no_grad():
                base_outputs = self.base_model(**input_ids, output_hidden_states=True)
                # Use the last hidden layer by default
                base_hidden_states.append(base_outputs.hidden_states[-1].detach().cpu())

            # Process with verified model
            with torch.no_grad():
                verified_outputs = self.verified_model(**input_ids, output_hidden_states=True)
                # Use the last hidden layer by default
                verified_hidden_states.append(verified_outputs.hidden_states[-1].detach().cpu())

        # Analyze the representations
        representations = self._compare_representations(base_hidden_states, verified_hidden_states, test_prompts)

        # Store the results
        self.results["representational_analysis"] = representations

        return representations

    def _compare_representations(self, base_hidden_states, verified_hidden_states, test_prompts):
        """Compare representations between base and verified models"""
        logger.info("Comparing representations between models...")

        # Calculate comparison metrics
        metrics = {}

        # 1. Calculate cosine similarities between each prompt pair
        base_cosine_sims = []
        verified_cosine_sims = []

        for i in range(0, len(test_prompts), 2):
            if i+1 < len(base_hidden_states):
                # Flatten and normalize representations
                base_repr1 = base_hidden_states[i].mean(dim=1).squeeze()
                base_repr2 = base_hidden_states[i+1].mean(dim=1).squeeze()
                verified_repr1 = verified_hidden_states[i].mean(dim=1).squeeze()
                verified_repr2 = verified_hidden_states[i+1].mean(dim=1).squeeze()

                # Calculate cosine similarity
                base_sim = F.cosine_similarity(base_repr1, base_repr2, dim=0).item()
                verified_sim = F.cosine_similarity(verified_repr1, verified_repr2, dim=0).item()

                base_cosine_sims.append(base_sim)
                verified_cosine_sims.append(verified_sim)

        metrics["base_cosine_similarities"] = base_cosine_sims
        metrics["verified_cosine_similarities"] = verified_cosine_sims

        # Calculate average similarities
        metrics["avg_base_similarity"] = sum(base_cosine_sims) / len(base_cosine_sims) if base_cosine_sims else 0
        metrics["avg_verified_similarity"] = sum(verified_cosine_sims) / len(verified_cosine_sims) if verified_cosine_sims else 0

        # Create meaningful output about what this means
        if metrics["avg_verified_similarity"] < metrics["avg_base_similarity"]:
            metrics["interpretation"] = "The verified model creates more distinct representations for truth vs. falsehood compared to the base model."
        else:
            metrics["interpretation"] = "The verified model creates more similar representations for truth vs. falsehood compared to the base model."

        # Log findings
        logger.info(f"Average similarity in base model: {metrics['avg_base_similarity']:.4f}")
        logger.info(f"Average similarity in verified model: {metrics['avg_verified_similarity']:.4f}")
        logger.info(f"Interpretation: {metrics['interpretation']}")

        # Plot the similarities
        self._plot_representation_similarities(base_cosine_sims, verified_cosine_sims, test_prompts)

        # Visualize the representations with dimensionality reduction
        if len(base_hidden_states) >= 4:
            self._visualize_representations(base_hidden_states, verified_hidden_states, test_prompts)

        return metrics

    def _plot_representation_similarities(self, base_sims, verified_sims, test_prompts):
        """Plot cosine similarities between paired prompt representations"""
        plt.figure(figsize=(12, 8))

        # Prepare data
        x = range(len(base_sims))

        # Plot similarities
        width = 0.35
        plt.bar([i - width/2 for i in x], base_sims, width, label='Base Model')
        plt.bar([i + width/2 for i in x], verified_sims, width, label='Verified Model')

        plt.xlabel('Prompt Pair')
        plt.ylabel('Cosine Similarity')
        plt.title('Representation Similarity Between Truth/Falsehood Pairs')

        # Add prompt pair labels if there are few enough
        pair_labels = []
        for i in range(0, len(test_prompts), 2):
            if i+1 < len(test_prompts):
                # Create abbreviated label
                p1 = test_prompts[i][:15] + "..." if len(test_prompts[i]) > 15 else test_prompts[i]
                p2 = test_prompts[i+1][:15] + "..." if len(test_prompts[i+1]) > 15 else test_prompts[i+1]
                pair_labels.append(f"{i//2+1}")

        plt.xticks(x, pair_labels)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add horizontal line at 0 and 1 for reference
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axhline(y=1, color='k', linestyle='-', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "representation_similarities.png"))
        plt.close()

        # Create a more detailed second plot for pairs if there are few enough
        if len(base_sims) <= 5:
            plt.figure(figsize=(14, 10))

            for i in range(len(base_sims)):
                plt.subplot(len(base_sims), 1, i+1)

                plt.bar(["Base Model", "Verified Model"], [base_sims[i], verified_sims[i]])

                # Get the corresponding prompts
                if 2*i+1 < len(test_prompts):
                    p1 = test_prompts[2*i]
                    p2 = test_prompts[2*i+1]
                    plt.title(f"Pair {i+1}: \"{p1}\" vs \"{p2}\"")

                plt.ylabel('Similarity')
                plt.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "detailed_similarities.png"))
            plt.close()

    def _visualize_representations(self, base_hidden_states, verified_hidden_states, test_prompts):
        """Visualize representations using PCA and t-SNE"""
        # Prepare data for dimensionality reduction
        base_reprs = [h.mean(dim=1).squeeze().numpy() for h in base_hidden_states]
        verified_reprs = [h.mean(dim=1).squeeze().numpy() for h in verified_hidden_states]

        # Make sure all representations have the same shape
        base_reprs = np.array([r.reshape(-1) for r in base_reprs])
        verified_reprs = np.array([r.reshape(-1) for r in verified_reprs])

        # Combine for joint visualization
        combined_reprs = np.vstack([base_reprs, verified_reprs])

        # Create labels
        labels = [f"Base-{i//2}-{'True' if i%2==0 else 'False'}" for i in range(len(base_reprs))]
        labels += [f"Verified-{i//2}-{'True' if i%2==0 else 'False'}" for i in range(len(verified_reprs))]

        # Use PCA for dimensionality reduction
        try:
            pca = PCA(n_components=2)
            reduced_reprs = pca.fit_transform(combined_reprs)

            # Plot PCA
            plt.figure(figsize=(10, 8))

            # Plot base model points
            for i in range(len(base_reprs)):
                point_type = 'o' if i % 2 == 0 else 'x'  # Circle for true, X for false
                plt.plot(reduced_reprs[i, 0], reduced_reprs[i, 1], point_type, 
                        color='blue', markersize=10, label='Base Model' if i == 0 else "")

            # Plot verified model points
            offset = len(base_reprs)
            for i in range(len(verified_reprs)):
                point_type = 'o' if i % 2 == 0 else 'x'  # Circle for true, X for false
                plt.plot(reduced_reprs[i+offset, 0], reduced_reprs[i+offset, 1], point_type, 
                        color='red', markersize=10, label='Verified Model' if i == 0 else "")

            # Add connecting lines between pairs (true/false)
            for i in range(0, len(base_reprs), 2):
                if i+1 < len(base_reprs):
                    # Base model pairs
                    plt.plot([reduced_reprs[i, 0], reduced_reprs[i+1, 0]], 
                            [reduced_reprs[i, 1], reduced_reprs[i+1, 1]], 
                            '-', color='blue', alpha=0.5)

                    # Verified model pairs
                    plt.plot([reduced_reprs[i+offset, 0], reduced_reprs[i+1+offset, 0]], 
                            [reduced_reprs[i+offset, 1], reduced_reprs[i+1+offset, 1]], 
                            '-', color='red', alpha=0.5)

            plt.title('PCA of Model Representations')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')

            # Add custom legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Base Model - True'),
                Line2D([0], [0], marker='x', color='blue', markersize=10, label='Base Model - False'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Verified Model - True'),
                Line2D([0], [0], marker='x', color='red', markersize=10, label='Verified Model - False')
            ]
            plt.legend(handles=legend_elements)

            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "pca_representations.png"))
            plt.close()

        except Exception as e:
            logger.warning(f"Error in PCA visualization: {e}")

        # Use t-SNE for dimensionality reduction
        try:
            tsne = TSNE(n_components=2, random_state=42)
            reduced_reprs = tsne.fit_transform(combined_reprs)

            # Plot t-SNE (same structure as PCA plot)
            plt.figure(figsize=(10, 8))

            # Plot base model points
            for i in range(len(base_reprs)):
                point_type = 'o' if i % 2 == 0 else 'x'  # Circle for true, X for false
                plt.plot(reduced_reprs[i, 0], reduced_reprs[i, 1], point_type, 
                        color='blue', markersize=10, label='Base Model' if i == 0 else "")

            # Plot verified model points
            offset = len(base_reprs)
            for i in range(len(verified_reprs)):
                point_type = 'o' if i % 2 == 0 else 'x'  # Circle for true, X for false
                plt.plot(reduced_reprs[i+offset, 0], reduced_reprs[i+offset, 1], point_type, 
                        color='red', markersize=10, label='Verified Model' if i == 0 else "")

            # Add connecting lines between pairs (true/false)
            for i in range(0, len(base_reprs), 2):
                if i+1 < len(base_reprs):
                    # Base model pairs
                    plt.plot([reduced_reprs[i, 0], reduced_reprs[i+1, 0]], 
                            [reduced_reprs[i, 1], reduced_reprs[i+1, 1]], 
                            '-', color='blue', alpha=0.5)

                    # Verified model pairs
                    plt.plot([reduced_reprs[i+offset, 0], reduced_reprs[i+1+offset, 0]], 
                            [reduced_reprs[i+offset, 1], reduced_reprs[i+1+offset, 1]], 
                            '-', color='red', alpha=0.5)

            plt.title('t-SNE of Model Representations')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')

            # Add custom legend (same as PCA plot)
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Base Model - True'),
                Line2D([0], [0], marker='x', color='blue', markersize=10, label='Base Model - False'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Verified Model - True'),
                Line2D([0], [0], marker='x', color='red', markersize=10, label='Verified Model - False')
            ]
            plt.legend(handles=legend_elements)

            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "tsne_representations.png"))
            plt.close()

        except Exception as e:
            logger.warning(f"Error in t-SNE visualization: {e}")

    def run_ablation_studies(self, ablation_models=None):
        """
        Run ablation studies to determine the impact of different components

        Args:
            ablation_models: Dictionary mapping ablation name to model path
                Example: {"no_cross_layer": "path/to/model_without_cross_layer"}
        """
        if ablation_models is None:
            logger.warning("No ablation models provided. Skipping ablation studies.")
            return None

        if self.verified_model is None:
            self.load_models()

        logger.info("Running ablation studies...")

        # Load ablation models
        loaded_ablations = {}
        for name, path in ablation_models.items():
            logger.info(f"Loading ablation model '{name}' from {path}")
            try:
                # Try loading with verification wrapper if possible
                try:
                    from enhanced_verification import load_bayesian_verification_model
                    model = load_bayesian_verification_model(path).to(self.device)
                except:
                    # Fallback to standard loading
                    model = AutoModelForCausalLM.from_pretrained(path).to(self.device)

                loaded_ablations[name] = model
                logger.info(f"Successfully loaded ablation model '{name}'")
            except Exception as e:
                logger.error(f"Error loading ablation model '{name}': {e}")

        # If no ablation models were successfully loaded, return
        if not loaded_ablations:
            logger.warning("No ablation models could be loaded. Skipping ablation studies.")
            return None

        # Run the same evaluations on all ablation models
        ablation_results = {}

        # 1. Factual knowledge evaluation
        ablation_results["factual_knowledge"] = self._run_factual_ablation(loaded_ablations)

        # 2. Verification activation analysis
        ablation_results["verification_activations"] = self._run_activation_ablation(loaded_ablations)

        # Store results
        self.results["ablation_studies"] = ablation_results

        return ablation_results

    def _run_factual_ablation(self, ablation_models, sample_size=20):
        """Run factual knowledge evaluation on ablation models"""
        logger.info("Running factual knowledge ablation study...")

        # Load a small test set
        test_set = self._create_minimal_factual_test()
        if sample_size < len(test_set):
            test_set = test_set[:sample_size]

        # Track results for each model
        results = {
            "base_model": {"correct": 0, "total": 0, "accuracy": 0},
            "verified_model": {"correct": 0, "total": 0, "accuracy": 0},
        }

        # Add ablation models to results
        for name in ablation_models:
            results[name] = {"correct": 0, "total": 0, "accuracy": 0}

        # Process each example
        for i, example in enumerate(tqdm(test_set, desc="Evaluating ablation models")):
            question, choices, correct_idx = example["question"], example["choices"], example["correct_idx"]

            # Format the input for the model
            input_text = f"Question: {question}\n\nChoices:\n"
            for j, choice in enumerate(choices):
                input_text += f"{j+1}. {choice}\n"
            input_text += "\nProvide the number of the correct answer:"

            # Process with base model
            base_output = self._get_model_answer(self.base_model, input_text)
            base_pred = self._extract_answer_idx(base_output, len(choices))
            base_correct = (base_pred == correct_idx)
            results["base_model"]["correct"] += int(base_correct)
            results["base_model"]["total"] += 1

            # Process with verified model
            verified_output = self._get_model_answer(self.verified_model, input_text)
            verified_pred = self._extract_answer_idx(verified_output, len(choices))
            verified_correct = (verified_pred == correct_idx)
            results["verified_model"]["correct"] += int(verified_correct)
            results["verified_model"]["total"] += 1

            # Process with ablation models
            for name, model in ablation_models.items():
                ablation_output = self._get_model_answer(model, input_text)
                ablation_pred = self._extract_answer_idx(ablation_output, len(choices))
                ablation_correct = (ablation_pred == correct_idx)
                results[name]["correct"] += int(ablation_correct)
                results[name]["total"] += 1

        # Calculate final accuracies
        for model_name in results:
            if results[model_name]["total"] > 0:
                results[model_name]["accuracy"] = results[model_name]["correct"] / results[model_name]["total"]

        # Plot results
        self._plot_ablation_results(results, "factual_knowledge")

        return results

    def _run_activation_ablation(self, ablation_models, num_samples=5):
        """Compare verification activations across ablation models"""
        logger.info("Running verification activation ablation study...")

        # Default test prompts
        test_prompts = [
            "The capital of France is Berlin.",
            "The tallest mountain in the world is Mount Everest.",
            "The human heart has three chambers.",
            "The sun revolves around the Earth.",
            "Water boils at 100 degrees Celsius at sea level."
        ][:num_samples]

        # Track activation metrics for each model
        activation_results = {
            "verified_model": {
                "confidence_scores": [],
                "cross_layer_consistency": [],
                "hidden_state_changes": []
            }
        }

        # Add ablation models
        for name in ablation_models:
            activation_results[name] = {
                "confidence_scores": [],
                "cross_layer_consistency": [],
                "hidden_state_changes": []
            }

        # Process each prompt
        for prompt in test_prompts:
            # Get metrics for verified model
            metrics = self._get_activation_metrics(self.verified_model, prompt)

            # Store results
            if metrics["confidence_scores"]:
                activation_results["verified_model"]["confidence_scores"].append(metrics["confidence_scores"])

            if metrics["cross_layer_consistency"] is not None:
                activation_results["verified_model"]["cross_layer_consistency"].append(metrics["cross_layer_consistency"])

            if metrics["hidden_state_changes"]:
                activation_results["verified_model"]["hidden_state_changes"].append(metrics["hidden_state_changes"])

            # Process ablation models
            for name, model in ablation_models.items():
                # Get metrics for this ablation model
                ablation_metrics = self._get_activation_metrics(model, prompt)

                # Store results
                if ablation_metrics["confidence_scores"]:
                    activation_results[name]["confidence_scores"].append(ablation_metrics["confidence_scores"])

                if ablation_metrics["cross_layer_consistency"] is not None:
                    activation_results[name]["cross_layer_consistency"].append(ablation_metrics["cross_layer_consistency"])

                if ablation_metrics["hidden_state_changes"]:
                    activation_results[name]["hidden_state_changes"].append(ablation_metrics["hidden_state_changes"])

        # Calculate averages
        for model_name in activation_results:
            model_results = activation_results[model_name]

            # Average confidence scores
            if model_results["confidence_scores"]:
                # Average across prompts and layers
                avg_confidence = np.mean([np.mean(scores) for scores in model_results["confidence_scores"]])
                model_results["avg_confidence"] = avg_confidence

            # Average cross-layer consistency
            if model_results["cross_layer_consistency"]:
                avg_consistency = np.mean(model_results["cross_layer_consistency"])
                model_results["avg_consistency"] = avg_consistency

            # Average hidden state changes
            if model_results["hidden_state_changes"]:
                avg_changes = np.mean([np.mean(changes) for changes in model_results["hidden_state_changes"]])
                model_results["avg_hidden_state_change"] = avg_changes

        # Plot comparisons
        self._plot_ablation_activations(activation_results)

        return activation_results

    def _get_activation_metrics(self, model, prompt):
        """Get verification activation metrics for a model and prompt"""
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Initialize metrics
        metrics = {
            "confidence_scores": [],
            "cross_layer_consistency": None,
            "hidden_state_changes": []
        }

        # Forward pass
        with torch.no_grad():
            # Check if model has verification attributes
            has_verification_adapters = hasattr(model, 'verification_adapters')

            # Store original hooks
            old_all_adapter_hidden_states = getattr(model, 'all_adapter_hidden_states', [])
            old_all_confidence_scores = getattr(model, 'all_confidence_scores', [])

            # Reset collections if possible
            if has_verification_adapters:
                model.all_adapter_hidden_states = []
                model.all_confidence_scores = []

            # Run forward pass
            outputs = model(**inputs, output_hidden_states=True)

            # Collect verification metrics
            verification_metrics = getattr(outputs, 'verification_metrics', {})

            # Check if we have access to confidence scores
            if hasattr(model, 'all_confidence_scores') and model.all_confidence_scores:
                metrics["confidence_scores"] = [score.mean().item() for score in model.all_confidence_scores]
            elif "layer_confidence_scores" in verification_metrics:
                metrics["confidence_scores"] = [score.mean().item() for score in verification_metrics["layer_confidence_scores"]]

            # Get cross-layer consistency if available
            if "cross_layer_consistency" in verification_metrics:
                metrics["cross_layer_consistency"] = verification_metrics["cross_layer_consistency"].mean().item()

            # Restore original hooks if needed
            if has_verification_adapters:
                model.all_adapter_hidden_states = old_all_adapter_hidden_states
                model.all_confidence_scores = old_all_confidence_scores

        return metrics

    def _plot_ablation_results(self, results, metric_name):
        """Plot comparison of results across ablation models"""
        plt.figure(figsize=(12, 8))

        # Prepare data for plotting
        models = []
        accuracies = []

        for model_name, result in results.items():
            if result["total"] > 0:  # Only include models with results
                models.append(model_name.replace("_", " ").title())
                accuracies.append(result["accuracy"] * 100)  # Convert to percentage

        # Create bar chart
        colors = ['blue', 'green'] + ['orange'] * (len(models) - 2)  # Highlight base and verified
        bars = plt.bar(models, accuracies, color=colors)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')

        plt.ylabel('Accuracy (%)')
        plt.title(f'Ablation Study: {metric_name.replace("_", " ").title()} Performance')
        plt.ylim(0, max(accuracies) * 1.2)  # Add some space above bars
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"ablation_{metric_name}.png"))
        plt.close()

    def _plot_ablation_activations(self, activation_results):
        """Plot comparison of activation metrics across ablation models"""
        # 1. Plot average confidence scores
        self._plot_single_ablation_metric(activation_results, "avg_confidence", "Confidence Score")

        # 2. Plot average cross-layer consistency
        self._plot_single_ablation_metric(activation_results, "avg_consistency", "Cross-Layer Consistency")

        # 3. Plot average hidden state changes
        self._plot_single_ablation_metric(activation_results, "avg_hidden_state_change", "Hidden State Change")

    def _plot_single_ablation_metric(self, results, metric_name, metric_title):
        """Plot a single metric across ablation models"""
        # Check if this metric exists for at least one model
        has_metric = any(metric_name in results[model] for model in results)
        if not has_metric:
            logger.warning(f"Metric {metric_name} not available for any model. Skipping plot.")
            return

        plt.figure(figsize=(10, 6))

        # Prepare data
        models = []
        values = []

        for model_name in results:
            if metric_name in results[model_name]:
                models.append(model_name.replace("_", " ").title())
                values.append(results[model_name][metric_name])

        if not values:
            logger.warning(f"No values available for {metric_name}. Skipping plot.")
            return

        # Create bar chart
        colors = ['green'] + ['orange'] * (len(models) - 1)  # Highlight verified model
        bars = plt.bar(models, values, color=colors)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height * 1.01,
                    f'{height:.3f}', ha='center', va='bottom')

        plt.ylabel(metric_title)
        plt.title(f'Ablation Study: {metric_title} Comparison')
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"ablation_{metric_name}.png"))
        plt.close()

    def run_layer_impact_analysis(self, adapter_locations=None):
        """
        Analyze the impact of different layers on verification performance

        Args:
            adapter_locations: List of adapter layer locations (if None, will try to extract from the model)
        """
        if self.verified_model is None:
            self.load_models()

        logger.info("Running layer impact analysis...")

        # Try to extract adapter locations if not provided
        if adapter_locations is None:
            if hasattr(self.verified_model, 'adapter_locations'):
                adapter_locations = self.verified_model.adapter_locations
                logger.info(f"Extracted adapter locations from model: {adapter_locations}")
            else:
                # Fallback to a guess based on the config
                num_layers = getattr(self.verified_model.config, 'num_hidden_layers', 
                                    getattr(self.verified_model.config, 'n_layer', 
                                           getattr(self.verified_model.config, 'num_layers', 12)))
                adapter_locations = list(range(2, num_layers, 3))
                logger.info(f"Using estimated adapter locations: {adapter_locations}")

        # Define test prompts for layer analysis
        test_prompts = [
            "The capital of France is Berlin.",
            "The tallest mountain in the world is K2.",
            "The human heart has three chambers.",
            "The Eiffel Tower is located in London.",
            "The primary colors are red, yellow, and blue."
        ]

        # Collect per-layer metrics
        layer_metrics = {
            "adapter_locations": adapter_locations,
            "confidence_scores": {},
            "correction_magnitudes": {},
            "impact_scores": {}
        }

        # Process each prompt
        for i, prompt in enumerate(test_prompts):
            logger.info(f"Analyzing layer impact for prompt {i+1}: '{prompt}'")

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Track metrics for this prompt
            prompt_metrics = self._measure_layer_impacts(inputs, adapter_locations)

            # Store metrics
            layer_metrics["confidence_scores"][f"prompt_{i+1}"] = prompt_metrics["confidence_scores"]
            layer_metrics["correction_magnitudes"][f"prompt_{i+1}"] = prompt_metrics["correction_magnitudes"]
            layer_metrics["impact_scores"][f"prompt_{i+1}"] = prompt_metrics["impact_scores"]

        # Calculate averages across prompts
        avg_confidence = []
        avg_corrections = []
        avg_impacts = []

        for layer_idx in range(len(adapter_locations)):
            # Collect values for this layer across all prompts
            layer_confidences = []
            layer_corrections = []
            layer_impacts = []

            for prompt_id in layer_metrics["confidence_scores"]:
                if layer_idx < len(layer_metrics["confidence_scores"][prompt_id]):
                    layer_confidences.append(layer_metrics["confidence_scores"][prompt_id][layer_idx])

                if layer_idx < len(layer_metrics["correction_magnitudes"][prompt_id]):
                    layer_corrections.append(layer_metrics["correction_magnitudes"][prompt_id][layer_idx])

                if layer_idx < len(layer_metrics["impact_scores"][prompt_id]):
                    layer_impacts.append(layer_metrics["impact_scores"][prompt_id][layer_idx])

            # Calculate averages
            avg_confidence.append(np.mean(layer_confidences) if layer_confidences else 0)
            avg_corrections.append(np.mean(layer_corrections) if layer_corrections else 0)
            avg_impacts.append(np.mean(layer_impacts) if layer_impacts else 0)

        # Store averages
        layer_metrics["avg_confidence"] = avg_confidence
        layer_metrics["avg_corrections"] = avg_corrections
        layer_metrics["avg_impacts"] = avg_impacts

        # Rank layers by impact
        layer_impact_ranking = sorted(
            [(idx, impact) for idx, impact in enumerate(avg_impacts)],
            key=lambda x: x[1],
            reverse=True
        )

        # Store layer rankings
        layer_metrics["layer_impact_ranking"] = [
            {"layer_idx": adapter_locations[idx], "impact_score": impact}
            for idx, impact in layer_impact_ranking
        ]

        # Log findings
        logger.info("Layer impact analysis results:")
        logger.info(f"  Most impactful layer: {layer_metrics['layer_impact_ranking'][0]['layer_idx']} (score: {layer_metrics['layer_impact_ranking'][0]['impact_score']:.4f})")
        logger.info(f"  Least impactful layer: {layer_metrics['layer_impact_ranking'][-1]['layer_idx']} (score: {layer_metrics['layer_impact_ranking'][-1]['impact_score']:.4f})")

        # Plot layer metrics
        self._plot_layer_metrics(layer_metrics)

        # Store results
        self.results["layer_impact"] = layer_metrics

        return layer_metrics

    def _measure_layer_impacts(self, inputs, adapter_locations):
        """Measure the impact of each verification layer on the given input"""
        # Initialize storage for metrics
        metrics = {
            "confidence_scores": [],
            "correction_magnitudes": [],
            "impact_scores": []
        }

        with torch.no_grad():
            # Check if verified model has necessary attributes
            has_adapters = hasattr(self.verified_model, 'verification_adapters')

            if not has_adapters:
                logger.warning("Verified model doesn't have verification_adapters attribute. Limited analysis possible.")
                return metrics

            # Store original hooks
            old_all_adapter_hidden_states = getattr(self.verified_model, 'all_adapter_hidden_states', [])
            old_all_confidence_scores = getattr(self.verified_model, 'all_confidence_scores', [])

            # Reset collections
            self.verified_model.all_adapter_hidden_states = []
            self.verified_model.all_confidence_scores = []

            # Run forward pass
            outputs = self.verified_model(**inputs, output_hidden_states=True)

            # Get all hidden states
            all_hidden_states = outputs.hidden_states

            # Calculate confidence scores
            if hasattr(self.verified_model, 'all_confidence_scores') and self.verified_model.all_confidence_scores:
                for score in self.verified_model.all_confidence_scores:
                    metrics["confidence_scores"].append(score.mean().item())

            # Calculate correction magnitudes
            if (hasattr(self.verified_model, 'all_adapter_hidden_states') and 
                len(self.verified_model.all_adapter_hidden_states) > 0 and
                len(all_hidden_states) >= max(adapter_locations) + 1):

                for i, (adapter_idx, corrected_state) in enumerate(zip(adapter_locations, self.verified_model.all_adapter_hidden_states)):
                    if adapter_idx < len(all_hidden_states):
                        # Get the input hidden state to the adapter (output from previous layer)
                        orig_state = all_hidden_states[adapter_idx]

                        # Calculate L2 norm of the difference
                        diff = (corrected_state - orig_state).norm().item()
                        # Normalize by tensor size
                        normalized_diff = diff / (orig_state.numel() ** 0.5)
                        metrics["correction_magnitudes"].append(normalized_diff)

            # Restore original hooks
            self.verified_model.all_adapter_hidden_states = old_all_adapter_hidden_states
            self.verified_model.all_confidence_scores = old_all_confidence_scores

            # Calculate impact scores as a combination of confidence and corrections
            # Lower confidence and higher correction magnitude = higher impact
            for i in range(min(len(metrics["confidence_scores"]), len(metrics["correction_magnitudes"]))):
                conf = metrics["confidence_scores"][i]
                corr = metrics["correction_magnitudes"][i]

                # Impact score: higher when confidence is low and corrections are high
                impact = (1 - conf) * corr
                metrics["impact_scores"].append(impact)

        return metrics

    def _plot_layer_metrics(self, layer_metrics):
        """Plot layer-specific metrics"""
        adapter_locations = layer_metrics["adapter_locations"]

        # 1. Plot average confidence scores by layer
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(adapter_locations)), layer_metrics["avg_confidence"])
        plt.xlabel('Adapter Index')
        plt.ylabel('Average Confidence Score')
        plt.title('Average Confidence Score by Layer')
        plt.xticks(range(len(adapter_locations)), [f"Layer {loc}" for loc in adapter_locations])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "layer_confidence.png"))
        plt.close()

        # 2. Plot average correction magnitudes by layer
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(adapter_locations)), layer_metrics["avg_corrections"])
        plt.xlabel('Adapter Index')
        plt.ylabel('Average Correction Magnitude')
        plt.title('Average Correction Magnitude by Layer')
        plt.xticks(range(len(adapter_locations)), [f"Layer {loc}" for loc in adapter_locations])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "layer_corrections.png"))
        plt.close()

        # 3. Plot average impact scores by layer
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(adapter_locations)), layer_metrics["avg_impacts"])
        plt.xlabel('Adapter Index')
        plt.ylabel('Average Impact Score')
        plt.title('Average Impact Score by Layer')
        plt.xticks(range(len(adapter_locations)), [f"Layer {loc}" for loc in adapter_locations])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "layer_impacts.png"))
        plt.close()

        # 4. Plot per-prompt layer impact heatmap
        plt.figure(figsize=(12, 8))

        # Prepare data for heatmap
        prompts = sorted(layer_metrics["impact_scores"].keys())
        heatmap_data = np.zeros((len(prompts), len(adapter_locations)))

        for i, prompt_id in enumerate(prompts):
            impact_scores = layer_metrics["impact_scores"][prompt_id]
            for j, score in enumerate(impact_scores):
                if j < heatmap_data.shape[1]:
                    heatmap_data[i, j] = score

        # Create heatmap
        sns.heatmap(heatmap_data, cmap="viridis", annot=True, fmt=".3f",
                    xticklabels=[f"Layer {loc}" for loc in adapter_locations],
                    yticklabels=prompts)

        plt.title('Layer Impact Scores by Prompt')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "layer_impact_heatmap.png"))
        plt.close()

    def run_consistency_evaluation(self, test_prompts=None, num_samples=5):
        """
        Evaluate whether the verification mechanism improves consistency in model outputs

        Args:
            test_prompts: List of prompts to test (if None, default prompts will be used)
            num_samples: Number of samples to generate per prompt
        """
        if self.base_model is None or self.verified_model is None:
            self.load_models()

        logger.info("Running consistency evaluation...")

        # Default test prompts if none provided
        if test_prompts is None:
            test_prompts = [
                "Explain the concept of gravity in simple terms.",
                "What are the main causes of climate change?",
                "Summarize the plot of Romeo and Juliet.",
                "Describe the process of photosynthesis.",
                "What are the key events that led to World War I?"
            ]

        # Track consistency metrics
        consistency_results = {
            "base_model": {
                "self_similarity": [],
                "output_entropy": [],
                "factual_consistency": []
            },
            "verified_model": {
                "self_similarity": [],
                "output_entropy": [],
                "factual_consistency": []
            }
        }

        # Process each prompt
        for prompt in test_prompts:
            logger.info(f"Evaluating consistency for prompt: {prompt}")

            # Generate multiple outputs for base model
            base_outputs = []
            for _ in range(num_samples):
                base_output = self._generate_with_sampling(self.base_model, prompt)
                base_outputs.append(base_output)

            # Generate multiple outputs for verified model
            verified_outputs = []
            for _ in range(num_samples):
                verified_output = self._generate_with_sampling(self.verified_model, prompt)
                verified_outputs.append(verified_output)

            # Calculate self-similarity for base model
            base_similarity = self._calculate_output_similarity(base_outputs)
            consistency_results["base_model"]["self_similarity"].append(base_similarity)

            # Calculate self-similarity for verified model
            verified_similarity = self._calculate_output_similarity(verified_outputs)
            consistency_results["verified_model"]["self_similarity"].append(verified_similarity)

            # Calculate output entropy
            base_entropy = self._calculate_output_entropy(base_outputs)
            verified_entropy = self._calculate_output_entropy(verified_outputs)

            consistency_results["base_model"]["output_entropy"].append(base_entropy)
            consistency_results["verified_model"]["output_entropy"].append(verified_entropy)

            # Calculate factual consistency (simple keyword-based approach)
            base_factual = self._estimate_factual_consistency(base_outputs)
            verified_factual = self._estimate_factual_consistency(verified_outputs)

            consistency_results["base_model"]["factual_consistency"].append(base_factual)
            consistency_results["verified_model"]["factual_consistency"].append(verified_factual)

        # Calculate averages
        for model in consistency_results:
            for metric in consistency_results[model]:
                values = consistency_results[model][metric]
                consistency_results[model][f"avg_{metric}"] = sum(values) / len(values) if values else 0

        # Log results
        logger.info("Consistency evaluation results:")
        logger.info(f"Base model avg self-similarity: {consistency_results['base_model']['avg_self_similarity']:.4f}")
        logger.info(f"Verified model avg self-similarity: {consistency_results['verified_model']['avg_self_similarity']:.4f}")
        logger.info(f"Base model avg output entropy: {consistency_results['base_model']['avg_output_entropy']:.4f}")
        logger.info(f"Verified model avg output entropy: {consistency_results['verified_model']['avg_output_entropy']:.4f}")
        logger.info(f"Base model avg factual consistency: {consistency_results['base_model']['avg_factual_consistency']:.4f}")
        logger.info(f"Verified model avg factual consistency: {consistency_results['verified_model']['avg_factual_consistency']:.4f}")

        # Plot results
        self._plot_consistency_results(consistency_results)

        # Store results
        self.results["consistency"] = consistency_results

        return consistency_results

    def _generate_with_sampling(self, model, prompt, max_tokens=100, temperature=0.7):
        """Generate text with sampling to test consistency"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.92,
                num_return_sequences=1
            )

        # Decode and return just the generated part
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_output[len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]

        return generated_text.strip()

    def _calculate_output_similarity(self, outputs):
        """Calculate average pairwise similarity between generated outputs"""
        if len(outputs) <= 1:
            return 1.0  # Perfect similarity with only one sample

        # Use tokenizer to get token IDs for more accurate comparison
        tokenized_outputs = [self.tokenizer.encode(output) for output in outputs]

        # Calculate Jaccard similarity between all pairs
        similarities = []
        for i in range(len(tokenized_outputs)):
            for j in range(i+1, len(tokenized_outputs)):
                set1 = set(tokenized_outputs[i])
                set2 = set(tokenized_outputs[j])

                if not set1 or not set2:
                    continue

                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))

                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)

        # Return average similarity
        return sum(similarities) / len(similarities) if similarities else 0

    def _calculate_output_entropy(self, outputs):
        """Calculate entropy of the output distribution"""
        if len(outputs) <= 1:
            return 0.0  # No entropy with only one sample

        # Tokenize all outputs
        tokenized_outputs = [self.tokenizer.encode(output) for output in outputs]

        # Flatten all tokens
        all_tokens = [token for output in tokenized_outputs for token in output]

        # Count token frequencies
        token_counts = {}
        for token in all_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        # Calculate entropy
        total_tokens = len(all_tokens)
        entropy = 0

        for token, count in token_counts.items():
            probability = count / total_tokens
            entropy -= probability * np.log2(probability)

        return entropy

    def _estimate_factual_consistency(self, outputs):
        """
        Estimate factual consistency across outputs
        Using a simple approach based on keyword consistency
        """
        if len(outputs) <= 1:
            return 1.0  # Perfect consistency with only one sample

        # Extract potential facts (names, numbers, dates, etc.)
        import re

        # Define patterns for potential facts
        patterns = [
            r'\b\d+\b',  # Numbers
            r'\b[A-Z][a-z]+\b',  # Capitalized words (potential names)
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Two capitalized words (potential names)
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # Dates in MM/DD/YYYY format
            r'\b\d{4}\b',  # Years
        ]

        # Extract facts from each output
        fact_sets = []
        for output in outputs:
            facts = set()
            for pattern in patterns:
                matches = re.findall(pattern, output)
                facts.update(matches)
            fact_sets.append(facts)

        # Calculate Jaccard similarity between all pairs of fact sets
        similarities = []
        for i in range(len(fact_sets)):
            for j in range(i+1, len(fact_sets)):
                set1 = fact_sets[i]
                set2 = fact_sets[j]

                if not set1 or not set2:
                    continue

                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))

                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)

        # Return average similarity
        return sum(similarities) / len(similarities) if similarities else 0

    def _plot_consistency_results(self, results):
        """Plot consistency evaluation results"""
        metrics = ["self_similarity", "output_entropy", "factual_consistency"]
        titles = ["Self-Similarity", "Output Entropy", "Factual Consistency"]

        # Create a figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            # Get values
            base_values = results["base_model"][metric]
            verified_values = results["verified_model"][metric]

            # Calculate averages
            base_avg = results["base_model"][f"avg_{metric}"]
            verified_avg = results["verified_model"][f"avg_{metric}"]

            # Plot individual values as scatter points
            axes[i].scatter(["Base Model"] * len(base_values), base_values, 
                         alpha=0.6, color='blue', label="Individual prompts")
            axes[i].scatter(["Verified Model"] * len(verified_values), verified_values, 
                         alpha=0.6, color='blue')

            # Plot averages as horizontal lines
            axes[i].plot(["Base Model", "Verified Model"], [base_avg, verified_avg], 
                      'r-', linewidth=2, label="Average")

            # Add text labels for averages
            axes[i].text("Base Model", base_avg, f' {base_avg:.3f}', 
                      verticalalignment='bottom', horizontalalignment='left')
            axes[i].text("Verified Model", verified_avg, f' {verified_avg:.3f}', 
                      verticalalignment='bottom', horizontalalignment='left')

            axes[i].set_title(title)
            axes[i].grid(True, linestyle='--', alpha=0.7)

            if i == 0:
                axes[i].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "consistency_results.png"))
        plt.close()

        # Additional plot: radar chart of the three metrics
        categories = ['Self-Similarity', 'Output Entropy', 'Factual Consistency']

        # Prepare data for radar chart
        base_values = [
            results["base_model"]["avg_self_similarity"],
            results["base_model"]["avg_output_entropy"] / 5,  # Normalize entropy to 0-1 range
            results["base_model"]["avg_factual_consistency"]
        ]

        verified_values = [
            results["verified_model"]["avg_self_similarity"],
            results["verified_model"]["avg_output_entropy"] / 5,  # Normalize entropy to 0-1 range
            results["verified_model"]["avg_factual_consistency"]
        ]

        # Set up radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        base_values += base_values[:1]
        verified_values += verified_values[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        ax.plot(angles, base_values, 'b-', linewidth=2, label='Base Model')
        ax.fill(angles, base_values, 'b', alpha=0.1)

        ax.plot(angles, verified_values, 'r-', linewidth=2, label='Verified Model')
        ax.fill(angles, verified_values, 'r', alpha=0.1)

        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        # Add legend
        ax.legend(loc='upper right')

        plt.title('Consistency Metrics Comparison')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "consistency_radar.png"))
        plt.close()

    def run_latent_dynamics_analysis(self, prompts=None, num_samples=3):
        """
        Analyze the dynamics of hidden states through the verification layers
        This directly tests the 'thinking in latent space' hypothesis

        Args:
            prompts: List of prompts to test (if None, default prompts will be used)
            num_samples: Number of samples if prompts is None
        """
        if self.base_model is None or self.verified_model is None:
            self.load_models()

        logger.info("Running latent dynamics analysis...")

        # Default test prompts if none provided
        if prompts is None:
            prompts = [
                "The capital of France is Berlin.",
                "The tallest mountain in the world is Mount Everest.",
                "The human heart has three chambers.",
                "The sun revolves around the Earth.",
                "Water boils at 100 degrees Celsius at sea level.",
            ][:num_samples]

        # Track latent dynamics
        dynamics_results = {
            "prompts": prompts,
            "base_trajectories": [],
            "verified_trajectories": [],
            "adapter_impact": []
        }

        # Process each prompt
        for prompt in prompts:
            logger.info(f"Analyzing latent dynamics for prompt: {prompt}")

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Track trajectories through layers
            base_trajectory, verified_trajectory, impact = self._trace_latent_trajectory(inputs)

            dynamics_results["base_trajectories"].append(base_trajectory)
            dynamics_results["verified_trajectories"].append(verified_trajectory)
            dynamics_results["adapter_impact"].append(impact)

        # Calculate average adapter impact
        avg_impact = np.zeros_like(dynamics_results["adapter_impact"][0])
        for impact in dynamics_results["adapter_impact"]:
            avg_impact += np.array(impact)
        avg_impact /= len(dynamics_results["adapter_impact"])
        dynamics_results["avg_adapter_impact"] = avg_impact.tolist()

        # Plot latent dynamics
        self._plot_latent_dynamics(dynamics_results)

        # Store results
        self.results["latent_dynamics"] = dynamics_results

        return dynamics_results

    def _trace_latent_trajectory(self, inputs):
        """Trace the trajectory of hidden states through layers"""
        # Process with base model to get baseline trajectory
        with torch.no_grad():
            base_outputs = self.base_model(**inputs, output_hidden_states=True)
            base_hidden_states = base_outputs.hidden_states

            # Compute distances between consecutive layers
            base_trajectory = []
            for i in range(1, len(base_hidden_states)):
                # Get consecutive hidden states
                prev_state = base_hidden_states[i-1]
                curr_state = base_hidden_states[i]

                # Calculate distance
                distance = (curr_state - prev_state).norm().item() / (prev_state.numel() ** 0.5)
                base_trajectory.append(distance)

        # Process with verified model
        with torch.no_grad():
            # Store original hooks
            old_all_adapter_hidden_states = getattr(self.verified_model, 'all_adapter_hidden_states', [])
            old_all_confidence_scores = getattr(self.verified_model, 'all_confidence_scores', [])

            # Reset collections
            self.verified_model.all_adapter_hidden_states = []
            self.verified_model.all_confidence_scores = []

            # Run forward pass
            verified_outputs = self.verified_model(**inputs, output_hidden_states=True)
            verified_hidden_states = verified_outputs.hidden_states

            # Compute distances between consecutive layers
            verified_trajectory = []
            for i in range(1, len(verified_hidden_states)):
                # Get consecutive hidden states
                prev_state = verified_hidden_states[i-1]
                curr_state = verified_hidden_states[i]

                # Calculate distance
                distance = (curr_state - prev_state).norm().item() / (prev_state.numel() ** 0.5)
                verified_trajectory.append(distance)

            # Calculate adapter impact as the difference in trajectories
            adapter_impact = []
            adapter_locations = getattr(self.verified_model, 'adapter_locations', [])

            # Default to empty impact if we can't determine it
            if not hasattr(self.verified_model, 'all_adapter_hidden_states') or not self.verified_model.all_adapter_hidden_states:
                adapter_impact = [0] * (len(verified_trajectory))
            else:
                # Calculate impact at each layer
                for i in range(len(verified_trajectory)):
                    if i in adapter_locations and i < len(base_trajectory):
                        # Calculate difference between base and verified trajectories
                        impact = abs(verified_trajectory[i] - base_trajectory[i])
                        adapter_impact.append(impact)
                    else:
                        adapter_impact.append(0)

            # Restore original hooks
            self.verified_model.all_adapter_hidden_states = old_all_adapter_hidden_states
            self.verified_model.all_confidence_scores = old_all_confidence_scores

        return base_trajectory, verified_trajectory, adapter_impact

    def _plot_latent_dynamics(self, dynamics_results):
        """Plot the latent dynamics results"""
        prompts = dynamics_results["prompts"]

        # 1. Plot trajectory comparisons for each prompt
        for i, (prompt, base_traj, verified_traj) in enumerate(zip(
            prompts, dynamics_results["base_trajectories"], dynamics_results["verified_trajectories"])):

            plt.figure(figsize=(12, 6))

            # Plot trajectories
            plt.plot(range(len(base_traj)), base_traj, 'b-', linewidth=2, label='Base Model')
            plt.plot(range(len(verified_traj)), verified_traj, 'r-', linewidth=2, label='Verified Model')

            # Highlight adapter locations if available
            adapter_locations = getattr(self.verified_model, 'adapter_locations', [])
            if adapter_locations:
                for loc in adapter_locations:
                    if loc < len(verified_traj):
                        plt.axvline(x=loc, color='g', linestyle='--', alpha=0.5)

            plt.xlabel('Layer')
            plt.ylabel('Normalized State Change')
            plt.title(f'Latent Trajectory: "{prompt}"')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"latent_trajectory_{i+1}.png"))
            plt.close()

        # 2. Plot average adapter impact
        plt.figure(figsize=(10, 6))
        avg_impact = dynamics_results["avg_adapter_impact"]

        plt.bar(range(len(avg_impact)), avg_impact)

        # Highlight adapter locations
        adapter_locations = getattr(self.verified_model, 'adapter_locations', [])
        if adapter_locations:
            for loc in adapter_locations:
                if loc < len(avg_impact):
                    plt.axvline(x=loc, color='r', linestyle='--', alpha=0.7)

        plt.xlabel('Layer')
        plt.ylabel('Average Adapter Impact')
        plt.title('Average Impact of Verification Adapters on Latent Dynamics')
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "avg_adapter_impact.png"))
        plt.close()

        # 3. Plot heatmap of adapter impact across prompts
        plt.figure(figsize=(12, 8))

        # Prepare data for heatmap
        impact_data = np.array(dynamics_results["adapter_impact"])

        # Create heatmap
        sns.heatmap(impact_data, cmap="viridis", 
                    xticklabels=[f"Layer {i}" for i in range(impact_data.shape[1])],
                    yticklabels=[f"Prompt {i+1}" for i in range(impact_data.shape[0])])

        plt.title('Adapter Impact Across Prompts and Layers')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "adapter_impact_heatmap.png"))
        plt.close()

    def run_token_flow_analysis(self, prompts=None, num_tokens=50):
        """
        Analyze how verification influences the token prediction flow during generation

        Args:
            prompts: List of prompts to test (if None, default prompts will be used)
            num_tokens: Number of tokens to generate and analyze
        """
        if self.base_model is None or self.verified_model is None:
            self.load_models()

        logger.info("Running token flow analysis...")

        # Default test prompts if none provided
        if prompts is None:
            prompts = [
                "The capital of France is ",
                "The tallest mountain in the world is ",
                "The human heart has ",
                "The Earth is "
            ]

        # Track token flow results
        flow_results = {
            "prompts": prompts,
            "base_tokens": [],
            "verified_tokens": [],
            "base_probabilities": [],
            "verified_probabilities": [],
            "top_changes": []
        }

        # Process each prompt
        for prompt in prompts:
            logger.info(f"Analyzing token flow for prompt: {prompt}")

            # Generate from base model
            base_tokens, base_probs = self._generate_with_probs(self.base_model, prompt, num_tokens)
            flow_results["base_tokens"].append(base_tokens)
            flow_results["base_probabilities"].append(base_probs)

            # Generate from verified model
            verified_tokens, verified_probs = self._generate_with_probs(self.verified_model, prompt, num_tokens)
            flow_results["verified_tokens"].append(verified_tokens)
            flow_results["verified_probabilities"].append(verified_probs)

            # Analyze significant changes in token probabilities
            top_changes = self._analyze_token_probability_changes(
                base_probs[0], verified_probs[0], base_tokens[0], verified_tokens[0])
            flow_results["top_changes"].append(top_changes)

        # Plot token flow results
        self._plot_token_flow_results(flow_results)

        # Store results
        self.results["token_flow"] = flow_results

        return flow_results

    def _generate_with_probs(self, model, prompt, num_tokens=10):
        """Generate tokens and return their probabilities"""
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = tokenized_prompt.input_ids.shape[1]

        generated_tokens = []
        token_probs = []

        # Generate tokens one by one to track probabilities
        current_input = tokenized_prompt.input_ids

        for _ in range(num_tokens):
            with torch.no_grad():
                outputs = model(current_input)
                next_token_logits = outputs.logits[:, -1, :].float()

                # Apply softmax to get probabilities
                next_token_probs = F.softmax(next_token_logits, dim=-1)

                # Get top token
                next_token = torch.argmax(next_token_probs, dim=-1).unsqueeze(0)

                # Store the probabilities of the top 10 tokens
                topk_probs, topk_indices = torch.topk(next_token_probs, 10, dim=-1)
                token_probs.append({
                    "topk_indices": topk_indices.cpu().numpy().tolist()[0],
                    "topk_probs": topk_probs.cpu().numpy().tolist()[0]
                })

                # Append the token to the current input
                current_input = torch.cat([current_input, next_token.T], dim=1)

                # Store the token
                token = next_token.item()
                generated_tokens.append(token)

        # Decode the generated tokens and return with probabilities
        decoded_tokens = self.tokenizer.decode(generated_tokens)

        return [decoded_tokens], [token_probs]

    def _analyze_token_probability_changes(self, base_probs, verified_probs, base_tokens, verified_tokens):
        """Analyze significant changes in token probabilities between models"""
        changes = []

        # Check if we have probability information
        if not base_probs or not verified_probs:
            return changes

        # Find minimum length to compare
        min_len = min(len(base_probs), len(verified_probs))

        for i in range(min_len):
            base_topk = base_probs[i]
            verified_topk = verified_probs[i]

            # Create dict of token probabilities for easy comparison
            base_dict = {idx: prob for idx, prob in zip(base_topk["topk_indices"], base_topk["topk_probs"])}
            verified_dict = {idx: prob for idx, prob in zip(verified_topk["topk_indices"], verified_topk["topk_probs"])}

            # Find tokens with significant probability changes
            all_tokens = set(base_dict.keys()).union(verified_dict.keys())

            for token in all_tokens:
                base_p = base_dict.get(token, 0)
                verified_p = verified_dict.get(token, 0)

                # Calculate absolute and relative change
                abs_change = verified_p - base_p
                rel_change = abs_change / (base_p + 1e-10)  # Avoid division by zero

                # Store if the change is significant
                if abs(abs_change) > 0.1 or abs(rel_change) > 0.5:
                    # Decode token
                    token_text = self.tokenizer.decode([token])

                    changes.append({
                        "position": i,
                        "token": token,
                        "token_text": token_text,
                        "base_prob": base_p,
                        "verified_prob": verified_p,
                        "abs_change": abs_change,
                        "rel_change": rel_change
                    })

        # Sort by absolute change magnitude
        changes.sort(key=lambda x: abs(x["abs_change"]), reverse=True)

        return changes[:10]  # Return top 10 changes

    def _plot_token_flow_results(self, flow_results):
        """Plot token flow analysis results"""
        prompts = flow_results["prompts"]

        # 1. Plot probability flow for first token for each prompt
        for i, prompt in enumerate(prompts):
            if i >= len(flow_results["base_probabilities"]) or i >= len(flow_results["verified_probabilities"]):
                continue

            base_probs = flow_results["base_probabilities"][i]
            verified_probs = flow_results["verified_probabilities"][i]

            if not base_probs or not verified_probs:
                continue

            # Get the first token's probabilities
            first_base = base_probs[0]
            first_verified = verified_probs[0]

            # Create bar chart comparing probabilities
            plt.figure(figsize=(12, 6))

            # Get top 5 tokens from each model
            top_tokens = set(first_base["topk_indices"][:5]).union(set(first_verified["topk_indices"][:5]))
            tokens = sorted(list(top_tokens))

            # Get probabilities for these tokens
            base_p = [first_base["topk_probs"][first_base["topk_indices"].index(t)] 
                     if t in first_base["topk_indices"] else 0 for t in tokens]
            verified_p = [first_verified["topk_probs"][first_verified["topk_indices"].index(t)] 
                         if t in first_verified["topk_indices"] else 0 for t in tokens]

            # Decode tokens for display
            token_labels = [self.tokenizer.decode([t]) for t in tokens]

            # Create bar chart
            x = range(len(tokens))
            width = 0.35

            plt.bar([i - width/2 for i in x], base_p, width, label='Base Model')
            plt.bar([i + width/2 for i in x], verified_p, width, label='Verified Model')

            plt.xlabel('Tokens')
            plt.ylabel('Probability')
            plt.title(f'Token Probabilities After Prompt: "{prompt}"')
            plt.xticks(x, token_labels)
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"token_probs_{i+1}.png"))
            plt.close()

        # 2. Plot top probability changes
        for i, (prompt, changes) in enumerate(zip(prompts, flow_results["top_changes"])):
            if not changes:
                continue

            plt.figure(figsize=(12, 8))

            # Display top 5 changes
            top5 = changes[:5]

            # Get tokens and changes
            tokens = [change["token_text"] for change in top5]
            base_probs = [change["base_prob"] for change in top5]
            verified_probs = [change["verified_prob"] for change in top5]

            # Create bar chart
            x = range(len(tokens))
            width = 0.35

            plt.bar([i - width/2 for i in x], base_probs, width, label='Base Model')
            plt.bar([i + width/2 for i in x], verified_probs, width, label='Verified Model')

            plt.xlabel('Tokens')
            plt.ylabel('Probability')
            plt.title(f'Largest Probability Changes: "{prompt}"')
            plt.xticks(x, tokens)
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"token_changes_{i+1}.png"))
            plt.close()

    def save_results(self, filename="evaluation_results.json"):
        """Save evaluation results to a JSON file"""
        # Convert results to a serializable format
        serializable_results = {}

        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, (list, dict, str, int, float, bool)) or v is None:
                        serializable_results[key][k] = v
                    else:
                        # Try to convert numpy arrays to lists
                        try:
                            serializable_results[key][k] = v.tolist()
                        except:
                            # If not convertible, use string representation
                            serializable_results[key][k] = str(v)
            else:
                # Top-level items
                if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
                    serializable_results[key] = value
                else:
                    try:
                        serializable_results[key] = value.tolist()
                    except:
                        serializable_results[key] = str(value)

        # Save to file
        with open(os.path.join(self.output_dir, filename), 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {os.path.join(self.output_dir, filename)}")

def run_full_evaluation_suite(
    base_model_name: str,
    verified_model_path: str,
    adapter_model_path: str = None,  # For parameter-matched baseline
    lora_model_path: str = None,     # For parameter-matched baseline
    mlp_model_path: str = None,      # For parameter-matched baseline
    output_dir: str = "evaluation_results",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Run the complete evaluation suite on a verification-enhanced model

    Args:
        base_model_name: Name or path of the original base model
        verified_model_path: Path to the model with verification components
        adapter_model_path: Path to adapter-only model for comparison (optional)
        lora_model_path: Path to LoRA model for comparison (optional)
        mlp_model_path: Path to MLP adapter model for comparison (optional)
        output_dir: Directory to save evaluation results
        device: Device to run evaluations on

    Returns:
        Dictionary with evaluation results
    """
    # Initialize evaluator
    evaluator = VerificationEvaluator(
        base_model_name=base_model_name,
        verified_model_path=verified_model_path,
        output_dir=output_dir,
        device=device
    )

    # Load models
    evaluator.load_models()

    # Load parameter-matched baselines if available
    if adapter_model_path or lora_model_path or mlp_model_path:
        evaluator.load_parameter_matched_baselines(
            adapter_model_path=adapter_model_path,
            lora_model_path=lora_model_path,
            mlp_model_path=mlp_model_path
        )

    # 1. Parameter count comparison
    param_counts = evaluator.compare_parameter_counts()

    # 2. Factual knowledge evaluation
    factual_results = evaluator.run_factual_knowledge_evaluation(sample_size=100)

    # 3. Verification activations analysis
    activation_results = evaluator.analyze_verification_activations(num_samples=5)

    # 4. Representational analysis
    rep_results = evaluator.run_representational_analysis(num_samples=5)

    # 5. Layer impact analysis
    layer_results = evaluator.run_layer_impact_analysis()

    # 6. Latent dynamics analysis
    dynamics_results = evaluator.run_latent_dynamics_analysis(num_samples=3)

    # 7. Token flow analysis
    flow_results = evaluator.run_token_flow_analysis()

    # 8. Consistency evaluation
    consistency_results = evaluator.run_consistency_evaluation(num_samples=5)

    # 9. Save results
    evaluator.save_results()

    return evaluator.results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate latent verification mechanism")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name or path")
    parser.add_argument("--verified_model", type=str, required=True, help="Verified model path")
    parser.add_argument("--adapter_model", type=str, default=None, help="Adapter-only model path (optional)")
    parser.add_argument("--lora_model", type=str, default=None, help="LoRA model path (optional)")
    parser.add_argument("--mlp_model", type=str, default=None, help="MLP adapter model path (optional)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")

    args = parser.parse_args()

    # Run the full evaluation suite
    results = run_full_evaluation_suite(
        base_model_name=args.base_model,
        verified_model_path=args.verified_model,
        adapter_model_path=args.adapter_model,
        lora_model_path=args.lora_model,
        mlp_model_path=args.mlp_model,
        output_dir=args.output_dir,
        device=args.device
    )
