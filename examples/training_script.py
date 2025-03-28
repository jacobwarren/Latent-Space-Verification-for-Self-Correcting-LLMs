from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import torch
from typing import Dict, List, Optional, Tuple

# Import verification components
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(parent_dir)

from latent_verification.enhanced_verification import BayesianVerificationAdapter
from latent_verification.latent_verification import LatentVerificationWrapper, CrossLayerVerifier


class VerificationTrainer(SFTTrainer):
    """
    Extension of SFTTrainer that adds verification-specific loss components
    """
    def __init__(
        self, 
        consistency_weight=0.01,
        confidence_regularization_weight=0.005,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.consistency_weight = consistency_weight
        self.confidence_regularization_weight = confidence_regularization_weight

        # Initialize tracking for verification metrics
        self.verification_metrics_history = {
            'cross_layer_consistency': [],
            'confidence_scores': []
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to include verification components.
        The 'num_items_in_batch' parameter is needed for compatibility with the latest Trainer.
        """
        # Forward pass with verification metrics
        outputs = model(**inputs)

        # Get the standard language modeling loss
        loss = outputs.loss

        # Add verification-specific losses if metrics are available
        if hasattr(outputs, "verification_metrics"):
            metrics = outputs.verification_metrics

            # Cross-layer consistency loss
            if "cross_layer_consistency" in metrics:
                consistency = metrics["cross_layer_consistency"]
                # Ensure tensor has dimensions for operations
                if consistency.dim() == 0:
                    consistency = consistency.view(1)
                # We want to maximize consistency (minimize 1 - consistency)
                consistency_loss = torch.mean(1 - consistency)
                loss += self.consistency_weight * consistency_loss

                # Track metric
                self.verification_metrics_history['cross_layer_consistency'].append(
                    consistency.detach().mean().item()
                )

            # Confidence regularization (prevent always-high or always-low confidence)
            if "layer_confidence_scores" in metrics:
                confidence_scores = metrics["layer_confidence_scores"]
                valid_scores = [conf for conf in confidence_scores if conf.requires_grad]

                if valid_scores:
                    # Convert means to tensors with dimensions before concatenating
                    confidence_means = [conf.mean().view(1) for conf in valid_scores]
                    avg_confidence = torch.mean(torch.cat(confidence_means))

                    # Penalize confidence scores that are too close to 0 or 1
                    safe_conf = torch.clamp(avg_confidence, 0.01, 0.99)  # Prevent extreme values
                    confidence_reg = -torch.log(safe_conf) - torch.log(1 - safe_conf)
                    confidence_reg = torch.clamp(confidence_reg, 0.0, 10.0)  # Limit maximum penalty

                    # confidence_reg = -torch.log(avg_confidence + 1e-10) - torch.log(1 - avg_confidence + 1e-10)
                    loss += self.confidence_regularization_weight * confidence_reg

                    # Track metric
                    self.verification_metrics_history['confidence_scores'].append(
                        avg_confidence.detach().item()
                    )

        return (loss, outputs) if return_outputs else loss

    def get_verification_metrics(self):
        """Returns the tracked verification metrics history"""
        return self.verification_metrics_history


def prepare_verification_model(model_name, verification_type="bayesian", adapter_locations=None):
    """
    Prepare a model with verification components
    """
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Get model configuration
    config = base_model.config

    # Determine hidden size
    hidden_size = getattr(config, 'hidden_size', 
                         getattr(config, 'd_model', 
                                getattr(config, 'n_embd', 768)))

    # Determine number of layers
    num_layers = getattr(config, 'num_hidden_layers', 
                        getattr(config, 'n_layer', 
                               getattr(config, 'num_layers', 12)))

    # Default adapter locations if not specified
    if adapter_locations is None:
        adapter_locations = list(range(2, num_layers, 3))

    print(f"Using adapter locations: {adapter_locations}")

    # Get device and dtype
    device = next(base_model.parameters()).device
    dtype = next(base_model.parameters()).dtype

    # Initialize verification adapters with correct dtype and device
    if verification_type == "bayesian":
        verification_adapters = torch.nn.ModuleDict({
            f"layer_{layer_idx}": BayesianVerificationAdapter(
                hidden_size=hidden_size,
                bottleneck_size=64,
            ).to(device=device, dtype=dtype)
            for layer_idx in adapter_locations
        })
    else:
        from latent_verification import VerificationAdapter
        verification_adapters = torch.nn.ModuleDict({
            f"layer_{layer_idx}": VerificationAdapter(
                hidden_size=hidden_size,
                bottleneck_size=64,
            ).to(device=device, dtype=dtype)
            for layer_idx in adapter_locations
        })

    # Create cross-layer verifier
    cross_layer_verifier = CrossLayerVerifier(
        hidden_size=hidden_size,
        num_layers=len(adapter_locations),
        bottleneck_size=64
    ).to(device=device, dtype=dtype)

    # Wrap the model with verification
    model = LatentVerificationWrapper(
        base_model=base_model,
        adapter_locations=adapter_locations,
        enable_cross_layer=True,
        freeze_base_model=True,
        verification_adapters=verification_adapters
    )

    # Move cross-layer verifier to the model
    model.cross_layer_verifier = cross_layer_verifier

    # Freeze base model parameters and unfreeze verification components
    for param in model.parameters():
        param.requires_grad = False

    for name, module in model.named_modules():
        if "verification_adapters" in name or "cross_layer_verifier" in name:
            for param_name, param in module.named_parameters():
                param.requires_grad = True

    return model


def train_verification_model(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    output_dir="verification_model",
    verification_type="bayesian",
    batch_size=1,
    gradient_accumulation_steps=16,
    num_epochs=1,
    max_samples=49664,
    learning_rate=2e-5,
    consistency_weight=0.01,
    confidence_regularization_weight=0.005
):
    """
    Train a verification model using TRL's SFTTrainer with custom loss
    """
    # Prepare model with verification components
    model = prepare_verification_model(
        model_name=model_name,
        verification_type=verification_type
    )

    max_seq_length = 4096

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, attn_implementation="flash_attention_2")
    if "Qwen" in model_name:
        tokenizer.pad_token = "<|image_pad|>"
        tokenizer.pad_token_id = 151655
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set up training arguments with modified saving behavior
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        do_eval=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=4,
        optim="adamw_torch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        fp16=False,
        bf16=True,  # Use BF16 precision
        max_steps=-1,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1,
        lr_scheduler_type="linear",
        max_seq_length=4096,
        dataset_text_field="text",
        report_to="none",  # Disable wandb/tensorboard logging
        gradient_checkpointing=False,  # Disable gradient checkpointing to avoid incompatibility
        save_safetensors=False,  # Important: Disable automatic safetensors saving by the trainer
    )

    # Load dataset
    from datasets import load_dataset
    raw_ds = load_dataset("open-r1/SYNTHETIC-1-SFT-Data-Code_decontaminated", "default", split=f"train[:{max_samples}]")

    # Define a function to process the dataset into the format needed for chat templates
    def process_smoltalk_dataset(examples):
        result = {"text": []}
        for i in range(len(examples["messages"])):
            messages = examples["messages"][i]
            if not isinstance(messages, list) or len(messages) == 0:
                # Skip empty or invalid messages
                result["text"].append("")
                continue

            # Process the conversation and apply the chat template
            try:
                formatted_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                result["text"].append(formatted_text)
            except Exception as e:
                print(f"Error formatting conversation: {e}")
                # Add empty string as fallback
                result["text"].append("")

        return result

    # Apply the processing function to the dataset
    ds = raw_ds.map(
        process_smoltalk_dataset,
        batched=True,
        batch_size=100,
        remove_columns=raw_ds.column_names,
    )

    # Filter out empty examples
    ds = ds.filter(lambda example: len(example["text"]) > 0)

    # Display a sample of processed data
    if len(ds) > 0:
        print("\nSample processed conversation:")
        print(ds[0]["text"][:200] + "..." if len(ds[0]["text"]) > 200 else ds[0]["text"])

    train_test_split = ds.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    def filter_by_max_length(ex):
        try:
            enc = tokenizer(ex["text"], return_tensors="pt")
            return enc["input_ids"].shape[1] <= max_seq_length
        except Exception as exc:
            print(f"Warning: Could not tokenize example, keeping by default. Error: {exc}")
            return True

    train_dataset_f = train_dataset.filter(filter_by_max_length)
    eval_dataset_f = eval_dataset.filter(filter_by_max_length)

    # Create our custom trainer
    trainer = VerificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_f,
        eval_dataset=eval_dataset_f,
        tokenizer=tokenizer,
        consistency_weight=consistency_weight,
        confidence_regularization_weight=confidence_regularization_weight,
        # No formatting_func needed since we've already processed the data
        max_seq_length=2048
    )

    # Modify the trainer's save method to use our custom save method
    original_save_model = trainer.save_model

    def custom_save_model(output_dir=None, _internal_call=False):
        """Custom save method that uses our wrapper's save_pretrained method"""
        if output_dir is None:
            output_dir = training_args.output_dir

        print(f"Using custom save method to save to {output_dir}/model")
        # Use the wrapper's save_pretrained method with safe_serialization=False
        model.save_pretrained(f"{output_dir}/model", safe_serialization=False)
        # Save tokenizer
        tokenizer.save_pretrained(f"{output_dir}/model")
        # Save training arguments
        trainer.save_state()

    # Replace the trainer's save method
    trainer.save_model = custom_save_model

    # Start training
    print(f"Starting training for {num_epochs} epochs...")
    trainer.train()

    # Get verification metrics
    metrics = trainer.get_verification_metrics()

    # Save model using our custom save method
    print("Saving final model...")
    model.save_pretrained('finetuned_model', safe_serialization=False)
    tokenizer.save_pretrained('finetuned_model')

    return model, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a self-validating LLM")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Model name or path")
    parser.add_argument("--verification_type", type=str, 
                        choices=["standard", "bayesian"], 
                        default="bayesian", 
                        help="Type of verification adapter")
    parser.add_argument("--output_dir", type=str, default="verification_model",
                        help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Number of steps to accumulate gradients")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max_samples", type=int, default=49664,
                        help="Maximum number of training samples to use")

    args = parser.parse_args()

    model, metrics = train_verification_model(
        model_name=args.model,
        output_dir=args.output_dir,
        verification_type=args.verification_type,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.epochs,
        max_samples=args.max_samples
    )

    print("\nTraining complete!")

    # Optionally, visualize the metrics
    import matplotlib.pyplot as plt

    if metrics.get('cross_layer_consistency', []):
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['cross_layer_consistency'])
        plt.title('Cross-Layer Consistency')
        plt.xlabel('Step')
        plt.ylabel('Consistency Score')
        plt.savefig(f"{args.output_dir}/consistency.png")
        plt.close()

    if metrics.get('confidence_scores', []):
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['confidence_scores'])
        plt.title('Average Confidence')
        plt.xlabel('Step')
        plt.ylabel('Confidence Score')
        plt.savefig(f"{args.output_dir}/confidence.png")
        plt.close()
