import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoTokenizer
import os
import sys
import torch.nn.functional as F

def analyze_hidden_state_changes(base_model, verified_model, tokenizer, true_false_pairs, output_dir="hidden_state_analysis"):
    """Analyze hidden state changes between base and verified models for true vs false statements"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Store results
    results = {
        "true_statements": [],
        "false_statements": [],
        "layer_impacts": {},
        "layer_distances": {}
    }

    # Get adapter locations
    adapter_locations = verified_model.adapter_locations if hasattr(verified_model, 'adapter_locations') else []

    for pair_idx, (true_stmt, false_stmt) in enumerate(true_false_pairs):
        print(f"Analyzing pair {pair_idx+1}: {true_stmt} vs {false_stmt}")

        # Process true statement
        true_inputs = tokenizer(true_stmt, return_tensors="pt").to(verified_model.device)

        # Get base model hidden states
        with torch.no_grad():
            base_true_outputs = base_model(**true_inputs, output_hidden_states=True)
            base_true_hidden = [h.detach().cpu() for h in base_true_outputs.hidden_states]

        # Get verified model hidden states
        with torch.no_grad():
            # Reset adapter tracking
            if hasattr(verified_model, 'all_adapter_hidden_states'):
                verified_model.all_adapter_hidden_states = []
            if hasattr(verified_model, 'all_confidence_scores'):
                verified_model.all_confidence_scores = []

            verified_true_outputs = verified_model(**true_inputs, output_hidden_states=True)
            verified_true_hidden = [h.detach().cpu() for h in verified_true_outputs.hidden_states]

            # Get confidence scores if available
            true_confidence = [conf.mean().item() for conf in verified_model.all_confidence_scores] if hasattr(verified_model, 'all_confidence_scores') and verified_model.all_confidence_scores else []

        # Process false statement
        false_inputs = tokenizer(false_stmt, return_tensors="pt").to(verified_model.device)

        # Get base model hidden states
        with torch.no_grad():
            base_false_outputs = base_model(**false_inputs, output_hidden_states=True)
            base_false_hidden = [h.detach().cpu() for h in base_false_outputs.hidden_states]

        # Get verified model hidden states
        with torch.no_grad():
            # Reset adapter tracking
            if hasattr(verified_model, 'all_adapter_hidden_states'):
                verified_model.all_adapter_hidden_states = []
            if hasattr(verified_model, 'all_confidence_scores'):
                verified_model.all_confidence_scores = []

            verified_false_outputs = verified_model(**false_inputs, output_hidden_states=True)
            verified_false_hidden = [h.detach().cpu() for h in verified_false_outputs.hidden_states]

            # Get confidence scores if available
            false_confidence = [conf.mean().item() for conf in verified_model.all_confidence_scores] if hasattr(verified_model, 'all_confidence_scores') and verified_model.all_confidence_scores else []

        # Calculate hidden state changes
        true_changes = []
        false_changes = []

        for layer_idx in range(min(len(base_true_hidden), len(verified_true_hidden))):
            # Get mean hidden state representation for each layer
            base_true_mean = base_true_hidden[layer_idx].mean(dim=1).squeeze()
            verified_true_mean = verified_true_hidden[layer_idx].mean(dim=1).squeeze()

            base_false_mean = base_false_hidden[layer_idx].mean(dim=1).squeeze()
            verified_false_mean = verified_false_hidden[layer_idx].mean(dim=1).squeeze()

            # Calculate L2 distance between base and verified
            true_diff = (verified_true_mean - base_true_mean).norm().item()
            false_diff = (verified_false_mean - base_false_mean).norm().item()

            # Normalize by hidden size
            true_diff = true_diff / np.sqrt(base_true_mean.numel())
            false_diff = false_diff / np.sqrt(base_false_mean.numel())

            true_changes.append(true_diff)
            false_changes.append(false_diff)

            # Store layer impacts
            layer_key = f"layer_{layer_idx}"
            if layer_key not in results["layer_impacts"]:
                results["layer_impacts"][layer_key] = {"true": [], "false": []}

            results["layer_impacts"][layer_key]["true"].append(true_diff)
            results["layer_impacts"][layer_key]["false"].append(false_diff)

        # Store individual pair results
        pair_results = {
            "true_statement": true_stmt,
            "false_statement": false_stmt,
            "true_changes": true_changes,
            "false_changes": false_changes,
            "true_confidence": true_confidence,
            "false_confidence": false_confidence
        }

        results["true_statements"].append(pair_results)
        results["false_statements"].append(pair_results)

        # Create visualization for this pair
        plt.figure(figsize=(12, 6))

        plt.plot(range(len(true_changes)), true_changes, 'g-', linewidth=2, label="True Statement")
        plt.plot(range(len(false_changes)), false_changes, 'r-', linewidth=2, label="False Statement")

        # Mark adapter locations if available
        for loc in adapter_locations:
            if loc < len(true_changes):
                plt.axvline(x=loc, color='b', linestyle='--', alpha=0.5)

        plt.xlabel('Layer')
        plt.ylabel('Hidden State Change Magnitude')
        plt.title(f'Hidden State Changes Across Layers\nTrue: "{true_stmt[:30]}..." vs False: "{false_stmt[:30]}..."')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"hidden_changes_pair_{pair_idx+1}.png"), dpi=300)
        plt.close()

    # Calculate average changes per layer for true vs false statements
    avg_true_changes = []
    avg_false_changes = []

    for layer in sorted(results["layer_impacts"].keys(), key=lambda x: int(x.split('_')[1])):
        avg_true = np.mean(results["layer_impacts"][layer]["true"])
        avg_false = np.mean(results["layer_impacts"][layer]["false"])

        avg_true_changes.append(avg_true)
        avg_false_changes.append(avg_false)

    # Create summary visualization
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.plot(range(len(avg_true_changes)), avg_true_changes, 'g-', linewidth=2, label="True Statements")
    plt.plot(range(len(avg_false_changes)), avg_false_changes, 'r-', linewidth=2, label="False Statements")

    # Mark adapter locations if available
    for loc in adapter_locations:
        if loc < len(avg_true_changes):
            plt.axvline(x=loc, color='b', linestyle='--', alpha=0.5)

    plt.xlabel('Layer')
    plt.ylabel('Average Hidden State Change')
    plt.title('Average Hidden State Change Magnitude by Layer')
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot the difference
    plt.subplot(1, 2, 2)
    diff = np.array(avg_false_changes) - np.array(avg_true_changes)

    plt.bar(range(len(diff)), diff, color=['green' if d < 0 else 'red' for d in diff])

    # Mark adapter locations if available
    for loc in adapter_locations:
        if loc < len(diff):
            plt.axvline(x=loc, color='b', linestyle='--', alpha=0.5)

    plt.xlabel('Layer')
    plt.ylabel('Difference (False - True)')
    plt.title('Difference in Hidden State Changes (False - True)')
    plt.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_hidden_changes.png"), dpi=300)
    plt.close()

    # Create PCA visualization of hidden states
    perform_pca_visualization(results, adapter_locations, output_dir)

    return results

def perform_pca_visualization(results, adapter_locations, output_dir):
    import os
    """Create PCA visualization of hidden states"""
    # Choose a middle adapter layer (if available)
    if adapter_locations:
        layer_idx = adapter_locations[len(adapter_locations) // 2]
    else:
        layer_idx = 12  # Default to a middle layer

    # Collect hidden states from true and false statements
    all_hidden_states = []
    labels = []

    for pair_idx, (true_result, false_result) in enumerate(zip(results["true_statements"], results["false_statements"])):
        # Skip if not enough data
        if len(true_result["true_changes"]) <= layer_idx or len(false_result["false_changes"]) <= layer_idx:
            continue

        # Add true statement data
        true_change = true_result["true_changes"][layer_idx]
        all_hidden_states.append([true_change, pair_idx, 1])  # 1 = true

        # Add false statement data
        false_change = false_result["false_changes"][layer_idx]
        all_hidden_states.append([false_change, pair_idx, 0])  # 0 = false

    # Create PCA visualization if we have enough data
    if len(all_hidden_states) >= 4:
        data = np.array(all_hidden_states)

        plt.figure(figsize=(10, 8))

        # Plot true vs false
        plt.scatter(data[data[:, 2] == 1, 0], data[data[:, 2] == 1, 1], color='green', label='True Statements')
        plt.scatter(data[data[:, 2] == 0, 0], data[data[:, 2] == 0, 1], color='red', label='False Statements')

        plt.xlabel('Hidden State Change Magnitude')
        plt.ylabel('Pair Index')
        plt.title(f'Hidden State Changes at Layer {layer_idx}')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"hidden_state_pca_layer_{layer_idx}.png"), dpi=300)
        plt.close()

def analyze_layer_specific_performance(base_model, verified_model, tokenizer, eval_dataset, output_dir="layer_specific_analysis"):
    """Analyze which verification layers contribute most to accuracy improvements"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Get adapter locations
    adapter_locations = verified_model.adapter_locations if hasattr(verified_model, 'adapter_locations') else []
    if not adapter_locations:
        print("No adapter locations found. Cannot perform layer-specific analysis.")
        return None

    # Store results
    results = {
        "base_accuracy": 0.0,
        "full_verified_accuracy": 0.0,
        "layer_specific_accuracy": {},
        "layer_impact_ranking": []
    }

    # Function to evaluate accuracy
    def evaluate_accuracy(model, dataset, prefix=""):
        correct = 0
        total = 0

        for item in dataset:
            question = item["question"]
            choices = item["choices"]
            correct_idx = item["correct_idx"]

            # Get model prediction
            inputs = tokenizer(question, return_tensors="pt").to(model.device)

            with torch.no_grad():
                # Generate prediction for each choice
                choice_scores = []

                for choice in choices:
                    choice_input = tokenizer(f"{question} {choice}", return_tensors="pt").to(model.device)
                    outputs = model(**choice_input)

                    # Use mean logit as score
                    score = outputs.logits.mean().item()
                    choice_scores.append(score)

                # Get highest scoring choice
                pred_idx = np.argmax(choice_scores)

                # Check if correct
                if pred_idx == correct_idx:
                    correct += 1

                total += 1

                if total % 10 == 0:
                    print(f"{prefix} Evaluated {total} examples. Current accuracy: {correct/total:.4f}")

        accuracy = correct / total if total > 0 else 0
        return accuracy

    # Evaluate base model
    print("Evaluating base model...")
    base_accuracy = evaluate_accuracy(base_model, eval_dataset, prefix="Base:")
    results["base_accuracy"] = base_accuracy

    # Evaluate full verified model
    print("Evaluating full verified model...")
    full_accuracy = evaluate_accuracy(verified_model, eval_dataset, prefix="Full:")
    results["full_verified_accuracy"] = full_accuracy

    # Create and evaluate single-layer models
    import copy

    print("Evaluating individual layer contributions...")
    for adapter_idx in adapter_locations:
        print(f"Testing adapter at layer {adapter_idx}...")

        # Create a version with only this adapter active
        # This is a simulation, we're not actually modifying the model structure

        # Evaluate with only this layer
        # We'll modify the forward pass behavior temporarily
        original_forward = verified_model.forward

        def modified_forward(*args, **kwargs):
            # Reset adapter tracking
            if hasattr(verified_model, 'all_adapter_hidden_states'):
                verified_model.all_adapter_hidden_states = []
            if hasattr(verified_model, 'all_confidence_scores'):
                verified_model.all_confidence_scores = []

            # Register hooks for each layer
            hooks = []

            def get_hook_fn(layer_idx):
                def hook_fn(module, input, output):
                    # Only apply the verification adapter at the current adapter_idx
                    if layer_idx == adapter_idx:
                        # Get the hidden states from the transformer layer output
                        if isinstance(output, tuple):
                            hidden_states = output[0]
                        else:
                            hidden_states = output

                        # Get adapter for this layer
                        adapter = verified_model.verification_adapters[f"layer_{layer_idx}"]

                        # Ensure hidden states have the right dtype
                        adapter_dtype = next(adapter.parameters()).dtype
                        hidden_states = hidden_states.to(dtype=adapter_dtype)

                        # Apply verification adapter
                        try:
                            corrected_states, confidence = adapter(hidden_states)

                            # Store for cross-layer verification
                            verified_model.all_adapter_hidden_states.append(corrected_states)
                            verified_model.all_confidence_scores.append(confidence)

                            # Return modified output
                            if isinstance(output, tuple):
                                return (corrected_states,) + output[1:]
                            else:
                                return corrected_states
                        except Exception as e:
                            print(f"Error in verification adapter at layer {layer_idx}: {e}")
                            return output

                    return output
                return hook_fn

            # Get transformer layers
            layers = verified_model._get_layers()

            # Register hooks on each layer
            for i, layer in enumerate(layers):
                hook = layer.register_forward_hook(get_hook_fn(i))
                hooks.append(hook)

            try:
                # Call the original forward method - MORE ROBUST VERSION:
                if len(args) > 0 and isinstance(args[0], dict):
                    # First arg is a dict, use it directly to avoid duplicating params
                    outputs = original_forward(**args[0])
                elif kwargs and 'input_ids' in kwargs:
                    # Use the kwargs directly
                    outputs = original_forward(**kwargs)
                else:
                    # Last resort - this might still cause the error in some cases
                    outputs = original_forward(*args, **kwargs)
            finally:
                # Remove hooks
                for hook in hooks:
                    hook.remove()

            return outputs

        # Replace forward method temporarily
        verified_model.forward = modified_forward.__get__(verified_model, type(verified_model))

        # Evaluate with only this layer active
        layer_accuracy = evaluate_accuracy(verified_model, eval_dataset[:20], prefix=f"Layer {adapter_idx}:")  # Using a subset for speed

        # Restore original forward method
        verified_model.forward = original_forward

        # Store results
        results["layer_specific_accuracy"][f"layer_{adapter_idx}"] = layer_accuracy

    # Rank layers by impact
    base_acc = results["base_accuracy"]
    impacts = [(layer, acc - base_acc) for layer, acc in results["layer_specific_accuracy"].items()]
    impacts.sort(key=lambda x: x[1], reverse=True)

    results["layer_impact_ranking"] = [{"layer_idx": int(layer.split('_')[1]), "impact_score": impact} for layer, impact in impacts]

    # Create visualizations
    plt.figure(figsize=(12, 6))

    # Prepare data for plotting
    layers = [int(layer.split('_')[1]) for layer in results["layer_specific_accuracy"].keys()]
    accuracies = list(results["layer_specific_accuracy"].values())

    # Sort by layer index
    idx_order = np.argsort(layers)
    layers = [layers[i] for i in idx_order]
    accuracies = [accuracies[i] for i in idx_order]

    # Plot layer-specific accuracies
    plt.bar(range(len(layers)), accuracies, color='blue', alpha=0.7)

    # Add reference lines
    plt.axhline(y=base_acc, color='red', linestyle='--', label=f'Base Accuracy: {base_acc:.4f}')
    plt.axhline(y=full_accuracy, color='green', linestyle='--', label=f'Full Verified Accuracy: {full_accuracy:.4f}')

    plt.xlabel('Adapter Layer')
    plt.ylabel('Accuracy')
    plt.title('Individual Layer Contribution to Accuracy')
    plt.xticks(range(len(layers)), layers)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layer_specific_accuracy.png"), dpi=300)
    plt.close()

    # Plot layer impact ranking
    plt.figure(figsize=(12, 6))

    # Prepare data
    ranked_layers = [item["layer_idx"] for item in results["layer_impact_ranking"]]
    impact_scores = [item["impact_score"] for item in results["layer_impact_ranking"]]

    # Plot impact scores
    bars = plt.bar(range(len(ranked_layers)), impact_scores, color='purple', alpha=0.7)

    # Add values on top of bars
    for bar, score in zip(bars, impact_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{score:.4f}',
                ha='center', va='bottom', rotation=0)

    plt.xlabel('Layer Rank')
    plt.ylabel('Impact Score (Accuracy Improvement)')
    plt.title('Verification Layers Ranked by Accuracy Impact')
    plt.xticks(range(len(ranked_layers)), ranked_layers)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layer_impact_ranking.png"), dpi=300)
    plt.close()

    return results

def analyze_token_probability_flow(base_model, verified_model, tokenizer, true_false_pairs, output_dir="token_probability_analysis"):
    """Analyze how verification shifts token probabilities"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Store results
    results = {
        "prompts": [],
        "top_tokens": [],
        "probability_shifts": []
    }

    for pair_idx, (true_stmt, false_stmt) in enumerate(true_false_pairs):
        # Analyze both statements
        for stmt_idx, stmt in enumerate([true_stmt, false_stmt]):
            stmt_type = "true" if stmt_idx == 0 else "false"
            print(f"Analyzing {stmt_type} statement: {stmt}")

            # Add prompt to results
            results["prompts"].append(stmt)

            # Tokenize input
            inputs = tokenizer(stmt, return_tensors="pt").to(verified_model.device)

            # Get predictions from base model
            with torch.no_grad():
                base_outputs = base_model(**inputs)
                base_logits = base_outputs.logits[:, -1, :]  # Logits for next token

                # Get top tokens
                base_probs = F.softmax(base_logits, dim=-1)
                base_top_probs, base_top_indices = torch.topk(base_probs, 30)  # Get top 30 tokens

                # Convert to lists
                base_top_indices = base_top_indices[0].tolist()
                base_top_probs = base_top_probs[0].tolist()

                # Decode tokens
                base_top_tokens = [tokenizer.decode([idx]) for idx in base_top_indices]

            # Get predictions from verified model
            with torch.no_grad():
                # Reset adapter tracking
                if hasattr(verified_model, 'all_adapter_hidden_states'):
                    verified_model.all_adapter_hidden_states = []
                if hasattr(verified_model, 'all_confidence_scores'):
                    verified_model.all_confidence_scores = []

                verified_outputs = verified_model(**inputs)
                verified_logits = verified_outputs.logits[:, -1, :]  # Logits for next token

                # Get top tokens
                verified_probs = F.softmax(verified_logits, dim=-1)
                verified_top_probs, verified_top_indices = torch.topk(verified_probs, 30)

                # Convert to lists
                verified_top_indices = verified_top_indices[0].tolist()
                verified_top_probs = verified_top_probs[0].tolist()

                # Decode tokens
                verified_top_tokens = [tokenizer.decode([idx]) for idx in verified_top_indices]

                # Get confidence scores if available
                confidences = []
                if hasattr(verified_model, 'all_confidence_scores') and verified_model.all_confidence_scores:
                    confidences = [conf.mean().item() for conf in verified_model.all_confidence_scores]

            # Store top tokens
            results["top_tokens"].append({
                "statement": stmt,
                "type": stmt_type,
                "base_tokens": list(zip(base_top_tokens, base_top_probs)),
                "verified_tokens": list(zip(verified_top_tokens, verified_top_probs)),
                "confidence_scores": confidences
            })

            # Calculate probability shifts for all tokens
            all_indices = set(base_top_indices + verified_top_indices)
            shifts = []

            for idx in all_indices:
                base_prob = base_probs[0, idx].item()
                verified_prob = verified_probs[0, idx].item()

                abs_shift = verified_prob - base_prob
                rel_shift = abs_shift / max(base_prob, 1e-10)  # Avoid division by zero

                token = tokenizer.decode([idx])

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

            # Store probability shifts
            results["probability_shifts"].append({
                "statement": stmt,
                "type": stmt_type,
                "shifts": shifts[:30]  # Keep top 30 shifts
            })

            # Create visualization for top tokens
            plt.figure(figsize=(12, 8))

            # Get top 5 tokens from each model
            all_top_tokens = set(base_top_tokens[:10] + verified_top_tokens[:10])
            tokens = sorted(list(all_top_tokens))[:15]  # Limit to 15 for readability

            # Get probabilities for these tokens
            base_probs_plot = []
            verified_probs_plot = []

            for token in tokens:
                # Find probability in base model
                base_prob = 0.0
                for t, p in zip(base_top_tokens, base_top_probs):
                    if t == token:
                        base_prob = p
                        break

                # Find probability in verified model
                verified_prob = 0.0
                for t, p in zip(verified_top_tokens, verified_top_probs):
                    if t == token:
                        verified_prob = p
                        break

                base_probs_plot.append(base_prob)
                verified_probs_plot.append(verified_prob)

            # Create bar chart
            x = range(len(tokens))
            width = 0.35

            plt.bar([i - width/2 for i in x], base_probs_plot, width, label='Base Model')
            plt.bar([i + width/2 for i in x], verified_probs_plot, width, label='Verified Model')

            plt.xlabel('Tokens')
            plt.ylabel('Probability')

            # Create abbreviated prompt for title
            prompt_abbr = stmt[:50] + "..." if len(stmt) > 50 else stmt
            plt.title(f'Token Probabilities for {stmt_type.title()} Statement:\n"{prompt_abbr}"')

            plt.xticks(x, tokens, rotation=45, ha='right')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"token_probs_{pair_idx+1}_{stmt_type}.png"), dpi=300)
            plt.close()

            # Create probability shift visualization
            plt.figure(figsize=(12, 8))

            # Take top 15 shifts by magnitude
            top_shifts = shifts[:15]

            # Get token labels and shifts
            token_labels = [shift["token"] for shift in top_shifts]
            abs_shifts = [shift["abs_shift"] for shift in top_shifts]

            # Create bar chart with color based on shift direction
            colors = ['green' if s > 0 else 'red' for s in abs_shifts]
            bars = plt.bar(range(len(top_shifts)), abs_shifts, color=colors)

            # Add token labels
            plt.xticks(range(len(top_shifts)), token_labels, rotation=45, ha='right')

            plt.xlabel('Tokens')
            plt.ylabel('Probability Shift')
            plt.title(f'Token Probability Shifts for {stmt_type.title()} Statement:\n"{prompt_abbr}"')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"prob_shifts_{pair_idx+1}_{stmt_type}.png"), dpi=300)
            plt.close()

            # Create comparative visualization
            plt.figure(figsize=(12, 10))

            # Top 10 largest positive and negative shifts
            positive_shifts = sorted([s for s in shifts if s["abs_shift"] > 0], key=lambda x: x["abs_shift"], reverse=True)[:5]
            negative_shifts = sorted([s for s in shifts if s["abs_shift"] < 0], key=lambda x: abs(x["abs_shift"]), reverse=True)[:5]

            combined_shifts = positive_shifts + negative_shifts

            # Extract data
            tokens = [s["token"] for s in combined_shifts]
            base_probs = [s["base_prob"] for s in combined_shifts]
            verified_probs = [s["verified_prob"] for s in combined_shifts]

            # Create scatter plot
            plt.subplot(2, 1, 1)
            for i, (token, base_p, verified_p) in enumerate(zip(tokens, base_probs, verified_probs)):
                plt.plot([base_p, verified_p], [i, i], 'o-', linewidth=1.5, 
                        color='green' if verified_p > base_p else 'red')

                # Add token labels
                plt.text(min(base_p, verified_p) - 0.02, i, token, ha='right', va='center')

                # Add probability values
                plt.text(base_p, i - 0.2, f"{base_p:.4f}", ha='center', va='top', fontsize=8)
                plt.text(verified_p, i + 0.2, f"{verified_p:.4f}", ha='center', va='bottom', fontsize=8)

            plt.xlabel('Probability')
            plt.title('Token Probability Shifts (Base → Verified)')
            plt.yticks([])
            plt.grid(axis='x', alpha=0.3)

            # Create bar chart of shifts
            plt.subplot(2, 1, 2)
            shifts_values = [s["abs_shift"] for s in combined_shifts]

            bars = plt.barh(range(len(combined_shifts)), shifts_values,
                          color=['green' if s > 0 else 'red' for s in shifts_values])

            plt.xlabel('Probability Shift')
            plt.title('Absolute Probability Shifts')
            plt.yticks(range(len(combined_shifts)), tokens)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

            plt.suptitle(f"Token Probability Analysis for {stmt_type.title()} Statement", fontsize=16)

            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.savefig(os.path.join(output_dir, f"token_analysis_{pair_idx+1}_{stmt_type}.png"), dpi=300)
            plt.close()

    # Create summary visualizations

    # 1. Average probability shift patterns by statement type
    plt.figure(figsize=(10, 6))

    # Process the shifts by statement type
    true_shifts = {}
    false_shifts = {}

    for shift_data in results["probability_shifts"]:
        stmt_type = shift_data["type"]
        for shift in shift_data["shifts"]:
            token = shift["token"]
            abs_shift = shift["abs_shift"]

            if stmt_type == "true":
                if token not in true_shifts:
                    true_shifts[token] = []
                true_shifts[token].append(abs_shift)
            else:
                if token not in false_shifts:
                    false_shifts[token] = []
                false_shifts[token].append(abs_shift)

    # Calculate average shifts
    true_avg_shifts = {token: np.mean(shifts) for token, shifts in true_shifts.items() if len(shifts) >= 2}
    false_avg_shifts = {token: np.mean(shifts) for token, shifts in false_shifts.items() if len(shifts) >= 2}

    # Get common tokens between true and false statements
    common_tokens = set(true_avg_shifts.keys()) & set(false_avg_shifts.keys())

    if common_tokens:
        # Extract data for plotting
        tokens = sorted(list(common_tokens))[:10]  # Take top 10 common tokens
        true_values = [true_avg_shifts[t] for t in tokens]
        false_values = [false_avg_shifts[t] for t in tokens]

        # Create bar chart
        x = range(len(tokens))
        width = 0.35

        plt.bar([i - width/2 for i in x], true_values, width, label='True Statements')
        plt.bar([i + width/2 for i in x], false_values, width, label='False Statements')

        plt.xlabel('Tokens')
        plt.ylabel('Average Probability Shift')
        plt.title('Average Token Probability Shifts by Statement Type')

        plt.xticks(x, tokens, rotation=45, ha='right')
        plt.legend()
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "avg_token_shifts_comparison.png"), dpi=300)
        plt.close()

    return results

def run_verification_analysis(base_model_name, verified_model_path, output_dir="verification_analysis"):
    """Run comprehensive analysis of verification mechanism"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load custom models
    parent_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(parent_dir)

    from latent_verification.enhanced_verification import load_bayesian_verification_model

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # Load verified model
    verified_model = load_bayesian_verification_model(verified_model_path)

    # Define test data
    true_false_pairs = [
        ("The capital of France is Paris.", "The capital of France is Berlin."),
        ("Water boils at 100 degrees Celsius at sea level.", "Water boils at 50 degrees Celsius at sea level."),
        ("The Earth orbits around the Sun.", "The Sun orbits around the Earth."),
        ("The human heart has four chambers.", "The human heart has three chambers."),
        ("Mount Everest is the tallest mountain in the world.", "K2 is the tallest mountain in the world.")
    ]

    # Create a small evaluation dataset for layer-specific tests
    eval_dataset = [
        {
            "question": "What is the capital of France?",
            "choices": ["Paris", "Berlin", "London", "Madrid"],
            "correct_idx": 0
        },
        {
            "question": "What is the boiling point of water at sea level?",
            "choices": ["50 degrees Celsius", "100 degrees Celsius", "0 degrees Celsius", "200 degrees Celsius"],
            "correct_idx": 1
        }
    ]

    # Run tests
    print("1. Analyzing hidden state changes...")
    hidden_results = analyze_hidden_state_changes(
        base_model, verified_model, tokenizer, true_false_pairs,
        output_dir=os.path.join(output_dir, "hidden_states")
    )

    print("2. Analyzing layer-specific performance...")
    layer_results = analyze_layer_specific_performance(
        base_model, verified_model, tokenizer, eval_dataset,
        output_dir=os.path.join(output_dir, "layer_performance")
    )

    print("3. Analyzing token probability flow...")
    token_results = analyze_token_probability_flow(
        base_model, verified_model, tokenizer, true_false_pairs[:2],  # Use subset for speed
        output_dir=os.path.join(output_dir, "token_flow")
    )

    return {
        "hidden_state_analysis": hidden_results,
        "layer_performance": layer_results,
        "token_flow": token_results
    }

# Run the analysis
if __name__ == "__main__":
    run_verification_analysis(
        base_model_name="Qwen/Qwen2.5-7B-Instruct",
        verified_model_path="../../model",
        output_dir="verification_analysis"
    )
