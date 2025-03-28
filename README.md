# Latent-Space Verification for Self-Correcting LLMs

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/abs/xxxx.xxxxx) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-orange.svg)](https://pytorch.org/)

Official implementation of **"Latent-Space Verification for Self-Correcting LLMs"**, which introduces a novel approach to enhance the factual reliability of Large Language Models through embedded verification mechanisms in latent space.

## Table of Contents
- [Overview](#overview)
- [Key Features and Contributions](#key-features-and-contributions)
- [Method](#method)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Experiments and Results](#experiments-and-results)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)

## Overview

Large Language Models (LLMs) excel at generating coherent text but often produce factual inaccuracies or hallucinations, limiting their reliability in critical applications. This repository introduces **Latent-Space Verification**, a novel approach that embeds verification mechanisms directly into the hidden layers of pre-trained transformers. By detecting and rectifying inaccuracies within latent representations before output generation, our method enhances factual accuracy while requiring minimal additional parameters.

<p align="center">
  <img src="assets/architecture.png" alt="Latent-Space Verification Architecture" width="80%"/>
</p>

## Key Features and Contributions

- **Latent-Space Verification**: Our mechanism enables models to identify and rectify inaccuracies within their latent representations before generating outputs.
- **Parameter-Efficient Implementation**: Using LoRA-style adapters with only a 0.08% parameter increase (6.3M parameters on a 7.6B model).
- **Significant Performance Gains**: Improves factual accuracy from 55.81% to 65.48% (nearly 10% absolute improvement).
- **Architectural Innovations**:
  - Residual Verification Networks with a mixture of experts approach
  - Uncertainty-weighted Bayesian corrections
  - Hierarchical verification across different abstraction levels
- **Extensive Analysis**: Direct evidence of "thinking in latent space" through visualization of hidden state dynamics, attention patterns, and more.

## Method

### Latent-Space Verification Architecture

Our method introduces verification modules into specific layers of a frozen pre-trained language model. These modules:

1. **Intercept hidden states** at strategic layers of the transformer
2. **Assess factual consistency** of the internal representations
3. **Apply confidence-weighted corrections** as needed
4. **Coordinate cross-layer verification** to ensure global consistency

### Key Components

- **Bayesian Verification Adapters**: Produce uncertainty-weighted corrections to hidden states
- **Residual Verification Networks**: Use mixture-of-experts approach with dynamic routing
- **Hierarchical Verification**: Implements verification at token, phrase, and semantic levels
- **Cross-Layer Verifier**: Ensures consistency of representations across different layers

## Installation

```bash
# Clone this repository
git clone https://github.com/jacobpaulwarren/latent-verification.git
cd latent-verification

# Create a conda environment
conda create -n latent-verify python=3.8
conda activate latent-verify

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Load a verification-enhanced model

```python
from latent_verification import load_verification_model

# Load a verification-enhanced model
model = load_verification_model("path/to/model")

# Generate text with verification enabled
outputs = model.generate(
    tokenizer("The capital of France is", return_tensors="pt").to(model.device),
    max_new_tokens=50
)
```

### Enhance your own model with verification

```python
from transformers import AutoModelForCausalLM
from latent_verification import create_verification_model

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("your/model/name")

# Add verification adapters (default: every 3rd layer)
verified_model = create_verification_model(
    base_model=base_model,
    adapter_locations=[2, 5, 8, 11, 14, 17, 20, 23],  # Customize adapter locations
    bottleneck_size=64,
    enable_cross_layer=True
)

# Fine-tune just the verification components
# (Base model parameters remain frozen)
# ... your fine-tuning code here ...

# Save the verification-enhanced model
verified_model.save_pretrained("path/to/save")
```

## Usage

### Training a Verification-Enhanced Model

For more advanced training with verification-specific loss functions:

```python
from transformers import TrainingArguments, Trainer
from latent_verification import VerificationLoss, create_verification_model

# Create model with verification components
model = create_verification_model("your/model/name")

# Create verification-aware trainer
class VerificationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        
        # Use verification metrics for loss calculation
        verification_metrics = getattr(outputs, "verification_metrics", None)
        
        # Combine with regular loss
        loss_fct = VerificationLoss(
            task_loss_fn=lambda o, t: o.loss,
            consistency_weight=0.1,
            confidence_regularization_weight=0.05
        )
        
        loss = loss_fct(outputs, inputs["labels"], verification_metrics)
        return (loss, outputs) if return_outputs else loss

# Training arguments
training_args = TrainingArguments(
    output_dir="./verification_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=3,
)

# Initialize trainer
trainer = VerificationTrainer(
    model=model,
    args=training_args,
    train_dataset=your_dataset,
)

# Train
trainer.train()
```

### Enhanced Verification Modules

For advanced verification capabilities:

```python
from enhanced_verification import (
    create_enhanced_verification_model,
    BayesianVerificationAdapter,
    HierarchicalVerifier,
    ResidualVerificationNetwork
)

# Create a model with Bayesian uncertainty estimation
bayesian_model = create_enhanced_verification_model(
    model_name_or_path="your/model/name",
    verification_type="bayesian"
)

# Create a model with hierarchical verification
hierarchical_model = create_enhanced_verification_model(
    model_name_or_path="your/model/name",
    verification_type="hierarchical"
)

# Create a model with residual verification networks
residual_model = create_enhanced_verification_model(
    model_name_or_path="your/model/name",
    verification_type="residual"
)

# Create a model with knowledge-grounded verification
knowledge_model = create_enhanced_verification_model(
    model_name_or_path="your/model/name",
    verification_type="knowledge"
)
```

### Analyzing Verification Dynamics

We provide specialized tools to visualize and analyze verification behavior:

```python
from confidence_analysis import ConfidenceAnalyzer
from embedding_dynamics import EmbeddingDynamicsAnalyzer
from latent_thinking import LatentThinkingAnalyzer

# Analyze verification confidence patterns
confidence_analyzer = ConfidenceAnalyzer(
    verified_model_path="path/to/verified/model",
    base_model_name="original/model/name",
    output_dir="confidence_analysis"
)
confidence_results = confidence_analyzer.analyze_truth_falsehood_confidence()

# Analyze embedding space dynamics
dynamics_analyzer = EmbeddingDynamicsAnalyzer(
    verified_model_path="path/to/verified/model",
    base_model_name="original/model/name",
    output_dir="embedding_dynamics"
)
dynamics_results = dynamics_analyzer.analyze_embedding_dynamics()

# Deep analysis of "thinking in latent space"
thinking_analyzer = LatentThinkingAnalyzer(
    base_model_name="original/model/name",
    verified_model_path="path/to/verified/model",
    output_dir="latent_thinking"
)
thinking_analyzer.analyze_hidden_state_trajectories()
thinking_analyzer.analyze_truth_falsehood_divergence()
thinking_analyzer.analyze_token_probability_flow()
```

## Experiments and Results

Our experiments demonstrate that latent-space verification significantly improves factual accuracy across multiple benchmarks:

### Factual Knowledge Accuracy

- **Base Model**: 55.81%
- **Verification-Enhanced Model**: 65.48%
- **Absolute Improvement**: 9.67%

With minimal parameter overhead: only a 0.08% parameter increase (6.3M additional parameters on a 7.6B model)

<p align="center">
  <img src="assets/factual_knowledge_accuracy.png" alt="Factual Knowledge Accuracy" width="60%"/>
</p>

### Evidence of Latent Thinking

Our analysis provides direct evidence for the "thinking in latent space" hypothesis:

<p align="center">
  <img src="assets/hidden_state_pca.png" alt="Hidden State PCA Visualization" width="60%"/>
</p>

- Verification layers create systematic transformations in embedding space
- Vector fields demonstrate directional correction patterns
- Confidence scores correlate with factual accuracy
- Token probability analysis shows shifts toward accurate completions

### Ablation Studies

Key findings from our ablation studies:

- **Cross-layer verification** is critical (37% reduction in improvement when disabled)
- **Learned confidence** outperforms fixed confidence thresholds by 43%
- **Layer placement** significantly impacts performance (middle layers contribute most)
- **Verification depth** affects different error types (early layers catch syntactic errors, late layers address semantic/factual issues)

## Repository Structure

```
latent-verification/
├── latent_verification/                # Core implementation
│   ├── __init__.py
│   ├── models.py                       # Core model components
│   ├── utils.py                        # Utility functions
│   ├── loss.py                         # Verification loss functions
│   └── detect_archi.py                 # Architecture detection utilities
├── enhanced_verification/              # Enhanced verification mechanisms
│   ├── __init__.py
│   ├── bayesian.py                     # Bayesian verification adapters
│   ├── hierarchical.py                 # Hierarchical verification
│   └── residual.py                     # Residual verification networks
├── analysis/                           # Analysis tools
│   ├── confidence_analysis.py          # Confidence pattern analysis
│   ├── embedding_dynamics.py           # Embedding space visualization
│   ├── latent_thinking.py              # Latent thinking analysis
│   └── verification_benchmark.py       # Benchmark generation
├── examples/                           # Usage examples
│   ├── train_verification.py           # Training script
│   ├── inference_example.py            # Inference example
│   └── analysis_example.py             # Analysis example
├── tests/                              # Unit tests
├── requirements.txt                    # Dependencies
├── setup.py                            # Package setup
└── README.md                           # This file
```

## Citation

If you use this work in your research, please cite our GitHub repository:

### BibTeX
```bibtex
@misc{warren2024latent,
  title={Latent-Space Verification for Self-Correcting LLMs},
  author={Warren, Jacob},
  year={2024},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/jacobpaulwarren/latent-verification}}
}
```

### If you upload to arXiv
If you upload your paper to arXiv, please use this citation instead:

```bibtex
@article{warren2024latent,
  title={Latent-Space Verification for Self-Correcting LLMs},
  author={Warren, Jacob},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

### APA
Warren, J. (2024). *Latent-Space Verification for Self-Correcting LLMs* [Computer software]. GitHub. https://github.com/jacobpaulwarren/latent-verification

### MLA
Warren, Jacob. "Latent-Space Verification for Self-Correcting LLMs." GitHub, 2024, github.com/jacobpaulwarren/latent-verification.

### IEEE
J. Warren, "Latent-Space Verification for Self-Correcting LLMs," GitHub, 2024. [Online]. Available: https://github.com/jacobpaulwarren/latent-verification

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This research was conducted independently.
- We thank the HuggingFace team for their transformer implementations.
- Experiments were conducted using Qwen 2.5 models at both 1.5B and 7B parameter scales.
