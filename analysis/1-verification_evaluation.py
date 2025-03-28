from evaluation_suite import VerificationEvaluator, run_full_evaluation_suite

# Basic Usage Example

# Setup paths
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Original base model
VERIFIED_MODEL = ""  # Relative location to model with verification mechanism
OUTPUT_DIR = "verification_eval_results"

# Run a specific evaluation
evaluator = VerificationEvaluator(
    base_model_name=BASE_MODEL,
    verified_model_path=VERIFIED_MODEL,
    output_dir=OUTPUT_DIR
)

# Load the models
evaluator.load_models()

# Analyze parameter count differences
param_results = evaluator.compare_parameter_counts()
print(f"Parameter efficiency ratio: {param_results['efficiency_ratio']:.6f}")

# Test factual knowledge
factual_results = evaluator.run_factual_knowledge_evaluation(sample_size=50)
base_accuracy = factual_results["base_model"]["accuracy"] 
verified_accuracy = factual_results["verified_model"]["accuracy"]
print(f"Base model accuracy: {base_accuracy:.4f}")
print(f"Verified model accuracy: {verified_accuracy:.4f}")
print(f"Improvement: {(verified_accuracy - base_accuracy) * 100:.2f}%")

# Analyze verification activations
activation_results = evaluator.analyze_verification_activations(num_samples=5)

# Save results and generate final report
evaluator.save_results()
evaluator.generate_final_report()

# Alternatively, run the complete evaluation suite
"""
results = run_full_evaluation_suite(
    base_model_name=BASE_MODEL,
    verified_model_path=VERIFIED_MODEL,
    output_dir=OUTPUT_DIR
)
"""

"""
# Advanced Usage with Ablation Studies

# Define ablation models for comparison
ablation_models = {
    "no_cross_layer": "path/to/model_without_cross_layer",
    "standard_adapter": "path/to/model_with_standard_adapter",
    "uncertainty_only": "path/to/model_with_uncertainty_only"
}

# Run ablation studies
ablation_results = evaluator.run_ablation_studies(ablation_models=ablation_models)

# Analyze which components contribute most to performance
for model_name, metrics in ablation_results["factual_knowledge"].items():
    if "accuracy" in metrics:
        print(f"{model_name} accuracy: {metrics['accuracy']:.4f}")
"""

