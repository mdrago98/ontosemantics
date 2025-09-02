import torch
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate_model(model, test_loader, config, device):
    """
    Comprehensive evaluation of relation extraction model

    Args:
        model: Trained relation extraction model
        test_loader: DataLoader for test dataset
        config: Configuration object with relation mappings
        device: Device to run evaluation on

    Returns:
        Dictionary with all evaluation metrics
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []
    total_loss = 0

    criterion = torch.nn.CrossEntropyLoss()

    print("Running evaluation...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["relation_label"].to(device)

            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            # Get predictions and probabilities
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Evaluated {batch_idx}/{len(test_loader)} batches")

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    # Calculate metrics
    results = calculate_metrics(all_labels, all_predictions, all_probabilities, config)
    results['loss'] = total_loss / len(test_loader)

    return results

def calculate_metrics(y_true, y_pred, y_proba, config):
    """
    Calculate comprehensive metrics for relation extraction

    Args:
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
        y_proba: Prediction probabilities (numpy array)
        config: Configuration with relation mappings

    Returns:
        Dictionary with all metrics
    """

    # Basic metrics
    accuracy = (y_true == y_pred).mean()

    # Micro-averaged metrics (overall performance)
    f1_micro = f1_score(y_true, y_pred, average='micro')
    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_micro = recall_score(y_true, y_pred, average='micro')

    # Macro-averaged metrics (average across classes)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')

    # Weighted metrics (weighted by class frequency)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Create detailed classification report
    target_names = [config.id_to_relation[i] for i in range(config.num_relations)]
    class_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    results = {
        # Overall metrics
        'accuracy': accuracy,
        'loss': 0,  # Will be set by evaluate_model

        # Micro-averaged (good for imbalanced datasets)
        'f1_micro': f1_micro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,

        # Macro-averaged (treats all classes equally)
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,

        # Weighted (accounts for class imbalance)
        'f1_weighted': f1_weighted,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,

        # Per-class details
        'per_class_metrics': {
            'precision': precision_per_class,
            'recall': recall_per_class,
            'f1': f1_per_class,
            'support': support_per_class
        },

        # Additional details
        'classification_report': class_report,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'true_labels': y_true,
        'probabilities': y_proba
    }

    return results

def print_evaluation_results(results, config):
    """
    Print formatted evaluation results

    Args:
        results: Results dictionary from calculate_metrics
        config: Configuration with relation mappings
    """

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    # Overall metrics
    print("\nOverall Performance:")
    print(f"  Accuracy:           {results['accuracy']:.4f}")
    print(f"  Loss:              {results['loss']:.4f}")

    print("\nMicro-averaged (Overall):")
    print(f"  Precision:         {results['precision_micro']:.4f}")
    print(f"  Recall:            {results['recall_micro']:.4f}")
    print(f"  F1-Score:          {results['f1_micro']:.4f}")

    print("\nMacro-averaged (Per-class average):")
    print(f"  Precision:         {results['precision_macro']:.4f}")
    print(f"  Recall:            {results['recall_macro']:.4f}")
    print(f"  F1-Score:          {results['f1_macro']:.4f}")

    print("\nWeighted (Class-balanced):")
    print(f"  Precision:         {results['precision_weighted']:.4f}")
    print(f"  Recall:            {results['recall_weighted']:.4f}")
    print(f"  F1-Score:          {results['f1_weighted']:.4f}")

    # Per-class breakdown
    print("\nPer-Class Performance:")
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 70)

    for i in range(config.num_relations):
        class_name = config.id_to_relation[i]
        precision = results['per_class_metrics']['precision'][i]
        recall = results['per_class_metrics']['recall'][i]
        f1 = results['per_class_metrics']['f1'][i]
        support = results['per_class_metrics']['support'][i]

        print(f"{class_name:<20} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")

def plot_confusion_matrix(results, config, save_path=None):
    """
    Plot confusion matrix

    Args:
        results: Results dictionary with confusion matrix
        config: Configuration with relation mappings
        save_path: Optional path to save the plot
    """

    plt.figure(figsize=(10, 8))

    # Get relation names
    class_names = [config.id_to_relation[i] for i in range(config.num_relations)]

    # Plot confusion matrix
    sns.heatmap(
        results['confusion_matrix'],
        annot=True,
        fmt='d',
        xticklabels=class_names,
        yticklabels=class_names,
        cmap='Blues'
    )

    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()

def analyze_errors(results, config, test_dataset=None, top_k=10):
    """
    Analyze the most common errors made by the model

    Args:
        results: Results dictionary
        config: Configuration with relation mappings
        test_dataset: Optional test dataset to get text examples
        top_k: Number of top errors to show
    """

    y_true = results['true_labels']
    y_pred = results['predictions']

    # Find all errors
    error_indices = np.where(y_true != y_pred)[0]

    if len(error_indices) == 0:
        print("🎉 No errors found! Perfect performance!")
        return

    # Analyze error patterns
    error_patterns = {}
    for idx in error_indices:
        true_label = y_true[idx]
        pred_label = y_pred[idx]

        true_class = config.id_to_relation[true_label]
        pred_class = config.id_to_relation[pred_label]

        error_pattern = f"{true_class} → {pred_class}"
        error_patterns[error_pattern] = error_patterns.get(error_pattern, 0) + 1

    # Sort by frequency
    sorted_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)

    print(f"\nERROR ANALYSIS ({len(error_indices)} total errors):")
    print("="*50)
    print(f"{'Error Pattern':<30} {'Count':<10} {'% of Errors':<12}")
    print("-" * 52)

    for pattern, count in sorted_errors[:top_k]:
        percentage = (count / len(error_indices)) * 100
        print(f"{pattern:<30} {count:<10} {percentage:<12.2f}%")

    return sorted_errors

def save_results_to_csv(results, config, filepath):
    """
    Save evaluation results to CSV file

    Args:
        results: Results dictionary
        config: Configuration with relation mappings
        filepath: Path to save CSV file
    """

    # Create per-class results DataFrame
    class_data = []
    for i in range(config.num_relations):
        class_data.append({
            'class': config.id_to_relation[i],
            'precision': results['per_class_metrics']['precision'][i],
            'recall': results['per_class_metrics']['recall'][i],
            'f1_score': results['per_class_metrics']['f1'][i],
            'support': results['per_class_metrics']['support'][i]
        })

    df = pd.DataFrame(class_data)

    # Add overall metrics as additional rows
    overall_data = [
        {'class': 'MICRO_AVG', 'precision': results['precision_micro'],
         'recall': results['recall_micro'], 'f1_score': results['f1_micro'], 'support': len(results['true_labels'])},
        {'class': 'MACRO_AVG', 'precision': results['precision_macro'],
         'recall': results['recall_macro'], 'f1_score': results['f1_macro'], 'support': len(results['true_labels'])},
        {'class': 'WEIGHTED_AVG', 'precision': results['precision_weighted'],
         'recall': results['recall_weighted'], 'f1_score': results['f1_weighted'], 'support': len(results['true_labels'])}
    ]

    df = pd.concat([df, pd.DataFrame(overall_data)], ignore_index=True)

    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")

# Complete evaluation workflow
def run_complete_evaluation(model, test_dataset_path, tokenizer, config, device, save_dir="./evaluation_results"):
    """
    Run complete evaluation workflow

    Args:
        model: Trained model
        test_dataset_path: Path to test dataset
        tokenizer: Tokenizer
        config: Configuration
        device: Device
        save_dir: Directory to save results
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Create test dataset and loader
    test_dataset = BioCRelationExtractionDataset(test_dataset_path, tokenizer, config)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                             collate_fn=collate_relation_batch)  # Use same collate function

    print(f"Test dataset size: {len(test_dataset)}")

    # Run evaluation
    results = evaluate_model(model, test_loader, config, device)

    # Print results
    print_evaluation_results(results, config)

    # Plot confusion matrix
    plot_confusion_matrix(results, config, f"{save_dir}/confusion_matrix.png")

    # Analyze errors
    error_patterns = analyze_errors(results, config, test_dataset)

    # Save results
    save_results_to_csv(results, config, f"{save_dir}/evaluation_results.csv")

    # Save detailed results as pickle
    import pickle
    with open(f"{save_dir}/detailed_results.pkl", 'wb') as f:
        pickle.dump(results, f)

    print(f"\n✅ Complete evaluation finished! Results saved to {save_dir}/")

    return results
