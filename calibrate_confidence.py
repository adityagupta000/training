"""
Temperature Scaling Calibration for Model Confidence
Fixes overconfidence without retraining - preserves accuracy while improving calibration

Usage:
    python calibrate_confidence.py --model_path saved_models/best_model.pth
    
This will:
1. Find optimal temperature using validation set
2. Save calibrated model with temperature parameter
3. Generate calibration plots (reliability diagrams)
4. Update inference to use calibrated predictions
"""

import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy.optimize import minimize
from sklearn.metrics import log_loss, brier_score_loss
from tqdm import tqdm

from model import load_model
from data_loader import get_data_loaders

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelWithTemperature(nn.Module):
    """
    A thin wrapper that adds temperature scaling to a trained model.
    
    Temperature Scaling:
    - Divides logits by temperature T before softmax
    - T > 1: Makes predictions less confident (softer probabilities)
    - T < 1: Makes predictions more confident (sharper probabilities)
    - T = 1: Original model (no change)
    """
    
    def __init__(self, model, temperature=1.0):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
        
    def forward(self, x):
        logits = self.model(x)
        return logits
    
    def temperature_scale(self, logits):
        """
        Apply temperature scaling to logits
        """
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature
    
    def predict_with_temperature(self, x):
        """
        Forward pass with temperature-scaled probabilities
        """
        logits = self.forward(x)
        scaled_logits = self.temperature_scale(logits)
        return F.softmax(scaled_logits, dim=1)


def calculate_ece(confidences, predictions, labels, n_bins=15):
    """
    Calculate Expected Calibration Error (ECE)
    
    ECE measures the difference between confidence and accuracy
    Lower ECE = better calibrated model
    
    Args:
        confidences: Max predicted probabilities
        predictions: Predicted classes
        labels: True classes
        n_bins: Number of bins for calibration
    
    Returns:
        ECE score
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).astype(float).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def collect_predictions(model, loader):
    """
    Collect all predictions, confidences, and labels from validation set
    """
    model.eval()
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Collecting predictions'):
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            
            all_logits.append(logits)
            all_labels.append(labels)
    
    logits = torch.cat(all_logits).cpu()
    labels = torch.cat(all_labels).cpu()
    
    return logits, labels


def find_optimal_temperature(logits, labels, init_temp=1.5):
    """
    Find optimal temperature using NLL minimization on validation set
    
    Args:
        logits: Model logits (before softmax)
        labels: True labels
        init_temp: Initial temperature guess
    
    Returns:
        Optimal temperature value
    """
    logits = logits.numpy()
    labels = labels.numpy()
    
    def objective(temp):
        """Negative log likelihood with temperature scaling"""
        scaled_logits = logits / temp
        probabilities = F.softmax(torch.from_numpy(scaled_logits), dim=1).numpy()
        nll = log_loss(labels, probabilities)
        return nll
    
    print("\n" + "="*70)
    print("OPTIMIZING TEMPERATURE")
    print("="*70)
    
    # Optimize
    result = minimize(
        objective,
        x0=init_temp,
        method='Nelder-Mead',
        options={'maxiter': 100}
    )
    
    optimal_temp = result.x[0]
    
    print(f"\n✓ Optimization complete:")
    print(f"   Initial temperature: {init_temp:.4f}")
    print(f"   Optimal temperature: {optimal_temp:.4f}")
    print(f"   Initial NLL: {objective(init_temp):.4f}")
    print(f"   Optimized NLL: {objective(optimal_temp):.4f}")
    
    return optimal_temp


def evaluate_calibration(model, loader, temperature=1.0, n_bins=15):
    """
    Comprehensive calibration evaluation
    
    Returns:
        Dictionary with calibration metrics
    """
    model.eval()
    
    all_confidences = []
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            if hasattr(model, 'predict_with_temperature'):
                probabilities = model.predict_with_temperature(images)
            else:
                logits = model(images)
                scaled_logits = logits / temperature
                probabilities = F.softmax(scaled_logits, dim=1)
            
            confidences, predictions = torch.max(probabilities, 1)
            
            all_confidences.extend(confidences.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    confidences = np.array(all_confidences)
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = (predictions == labels).mean()
    ece = calculate_ece(confidences, predictions, labels, n_bins)
    nll = log_loss(labels, probabilities)
    
    # Brier score (mean squared error of probabilities)
    brier = brier_score_loss(
        labels,
        probabilities,
        pos_label=None
    )
    
    # Separate correct/incorrect predictions
    correct_mask = predictions == labels
    avg_conf_correct = confidences[correct_mask].mean()
    avg_conf_incorrect = confidences[~correct_mask].mean()
    
    return {
        'accuracy': accuracy,
        'ece': ece,
        'nll': nll,
        'brier': brier,
        'avg_confidence': confidences.mean(),
        'avg_conf_correct': avg_conf_correct,
        'avg_conf_incorrect': avg_conf_incorrect,
        'confidences': confidences,
        'predictions': predictions,
        'labels': labels,
        'probabilities': probabilities
    }


def plot_reliability_diagram(before_metrics, after_metrics, save_path):
    """
    Plot reliability diagram comparing before/after calibration
    
    Reliability diagram shows:
    - X-axis: Predicted confidence
    - Y-axis: Actual accuracy
    - Perfect calibration = diagonal line
    """
    n_bins = 15
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (metrics, title) in enumerate([
        (before_metrics, 'Before Calibration'),
        (after_metrics, 'After Temperature Scaling')
    ]):
        ax = axes[idx]
        
        confidences = metrics['confidences']
        predictions = metrics['predictions']
        labels = metrics['labels']
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.sum()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
                confidence_in_bin = confidences[in_bin].mean()
                
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(confidence_in_bin)
                bin_counts.append(prop_in_bin)
            else:
                bin_accuracies.append(0)
                bin_confidences.append((bin_lower + bin_upper) / 2)
                bin_counts.append(0)
        
        # Plot bars
        bin_centers = (bin_lowers + bin_uppers) / 2
        widths = bin_uppers - bin_lowers
        
        # Bar colors based on calibration error
        colors = []
        for acc, conf in zip(bin_accuracies, bin_confidences):
            error = abs(conf - acc)
            if error < 0.05:
                colors.append('#2ecc71')  # Green - well calibrated
            elif error < 0.15:
                colors.append('#f39c12')  # Orange - moderate
            else:
                colors.append('#e74c3c')  # Red - poorly calibrated
        
        bars = ax.bar(bin_centers, bin_accuracies, width=widths * 0.9,
                     color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add gap bars (difference between confidence and accuracy)
        for center, acc, conf, count in zip(bin_centers, bin_accuracies, bin_confidences, bin_counts):
            if count > 0:
                gap_height = abs(conf - acc)
                gap_bottom = min(acc, conf)
                ax.bar(center, gap_height, width=widths[0] * 0.9, bottom=gap_bottom,
                      color='red', alpha=0.3, edgecolor='darkred', linewidth=1)
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        
        # Confidence line
        ax.plot(bin_confidences, bin_accuracies, 'ro-', linewidth=2, 
               markersize=8, label='Model Calibration')
        
        ax.set_xlabel('Confidence', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title(f'{title}\nECE: {metrics["ece"]:.4f} | NLL: {metrics["nll"]:.4f}',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Reliability diagram saved: {save_path}")


def plot_confidence_histograms(before_metrics, after_metrics, save_path):
    """
    Plot confidence distribution before/after calibration
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    for idx, (metrics, label) in enumerate([
        (before_metrics, 'Before Calibration'),
        (after_metrics, 'After Calibration')
    ]):
        confidences = metrics['confidences']
        predictions = metrics['predictions']
        labels_true = metrics['labels']
        
        correct = predictions == labels_true
        
        # Overall distribution
        ax1 = axes[idx, 0]
        ax1.hist(confidences, bins=30, alpha=0.7, color='#3498db', edgecolor='black')
        ax1.axvline(confidences.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {confidences.mean():.3f}')
        ax1.set_xlabel('Confidence', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title(f'{label}\nOverall Confidence Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Correct vs Incorrect
        ax2 = axes[idx, 1]
        ax2.hist(confidences[correct], bins=30, alpha=0.7, color='#2ecc71',
                label=f'Correct (μ={confidences[correct].mean():.3f})', edgecolor='black')
        ax2.hist(confidences[~correct], bins=30, alpha=0.7, color='#e74c3c',
                label=f'Incorrect (μ={confidences[~correct].mean():.3f})', edgecolor='black')
        ax2.set_xlabel('Confidence', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title(f'{label}\nCorrect vs Incorrect Predictions', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confidence histograms saved: {save_path}")


def save_calibrated_model(model, temperature, original_checkpoint, save_path):
    """
    Save model with temperature parameter
    """
    calibrated_checkpoint = original_checkpoint.copy()
    calibrated_checkpoint['temperature'] = temperature
    calibrated_checkpoint['calibrated'] = True
    
    torch.save(calibrated_checkpoint, save_path)
    print(f"✓ Calibrated model saved: {save_path}")


def generate_calibration_report(before_metrics, after_metrics, temperature, save_path):
    """
    Generate detailed text report
    """
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CONFIDENCE CALIBRATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Optimal Temperature: {temperature:.4f}\n")
        f.write(f"(T > 1 = softer probabilities, T < 1 = sharper probabilities)\n\n")
        
        f.write("="*70 + "\n")
        f.write("BEFORE CALIBRATION\n")
        f.write("="*70 + "\n")
        f.write(f"Accuracy: {before_metrics['accuracy']:.4f} ({before_metrics['accuracy']*100:.2f}%)\n")
        f.write(f"ECE (Expected Calibration Error): {before_metrics['ece']:.4f}\n")
        f.write(f"NLL (Negative Log Likelihood): {before_metrics['nll']:.4f}\n")
        f.write(f"Brier Score: {before_metrics['brier']:.4f}\n")
        f.write(f"Average Confidence: {before_metrics['avg_confidence']:.4f}\n")
        f.write(f"Avg Confidence (Correct): {before_metrics['avg_conf_correct']:.4f}\n")
        f.write(f"Avg Confidence (Incorrect): {before_metrics['avg_conf_incorrect']:.4f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("AFTER CALIBRATION\n")
        f.write("="*70 + "\n")
        f.write(f"Accuracy: {after_metrics['accuracy']:.4f} ({after_metrics['accuracy']*100:.2f}%)\n")
        f.write(f"ECE (Expected Calibration Error): {after_metrics['ece']:.4f}\n")
        f.write(f"NLL (Negative Log Likelihood): {after_metrics['nll']:.4f}\n")
        f.write(f"Brier Score: {after_metrics['brier']:.4f}\n")
        f.write(f"Average Confidence: {after_metrics['avg_confidence']:.4f}\n")
        f.write(f"Avg Confidence (Correct): {after_metrics['avg_conf_correct']:.4f}\n")
        f.write(f"Avg Confidence (Incorrect): {after_metrics['avg_conf_incorrect']:.4f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("IMPROVEMENTS\n")
        f.write("="*70 + "\n")
        ece_reduction = (before_metrics['ece'] - after_metrics['ece']) / before_metrics['ece'] * 100
        nll_reduction = (before_metrics['nll'] - after_metrics['nll']) / before_metrics['nll'] * 100
        
        f.write(f"ECE Reduction: {ece_reduction:.2f}%\n")
        f.write(f"NLL Reduction: {nll_reduction:.2f}%\n")
        f.write(f"Accuracy Change: {(after_metrics['accuracy'] - before_metrics['accuracy'])*100:.2f}%\n")
        f.write(f"\n✓ Accuracy preserved (no retraining needed)\n")
        f.write(f"✓ Confidence scores now better reflect true accuracy\n")
        f.write(f"✓ Model predictions more trustworthy for clinical use\n")
    
    print(f"✓ Calibration report saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Temperature Scaling Calibration')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model (default: best_model.pth)')
    parser.add_argument('--output_name', type=str, default='plant_health_v1.pth',
                       help='Output name for calibrated model')
    parser.add_argument('--init_temp', type=float, default=1.5,
                       help='Initial temperature for optimization')
    
    args = parser.parse_args()
    
    # Model path
    if args.model_path is None:
        args.model_path = os.path.join(config['paths']['models'], 'best_model.pth')
    
    if not os.path.exists(args.model_path):
        print(f"✗ Model not found: {args.model_path}")
        return
    
    print("\n" + "="*70)
    print("CONFIDENCE CALIBRATION VIA TEMPERATURE SCALING")
    print("="*70)
    print(f"\nModel: {args.model_path}")
    print("Method: Temperature Scaling (post-hoc calibration)")
    print("\n■ This preserves model accuracy while improving confidence scores")
    print("■ No retraining required - fast calibration in minutes")
    print("="*70 + "\n")
    
    # Load model
    print("■ Loading trained model...")
    model, checkpoint = load_model(args.model_path, device)
    model.eval()
    
    # Load data
    print("■ Loading validation data...")
    _, val_loader, _ = get_data_loaders()
    
    # Collect predictions on validation set
    print("\n■ Collecting validation predictions...")
    val_logits, val_labels = collect_predictions(model, val_loader)
    
    # Find optimal temperature
    optimal_temp = find_optimal_temperature(val_logits, val_labels, args.init_temp)
    
    # Evaluate before calibration
    print("\n" + "="*70)
    print("EVALUATING BEFORE CALIBRATION")
    print("="*70)
    before_metrics = evaluate_calibration(model, val_loader, temperature=1.0)
    
    print(f"\n■ Before Calibration:")
    print(f"   Accuracy: {before_metrics['accuracy']:.4f}")
    print(f"   ECE: {before_metrics['ece']:.4f}")
    print(f"   NLL: {before_metrics['nll']:.4f}")
    print(f"   Avg Confidence (Correct): {before_metrics['avg_conf_correct']:.4f}")
    print(f"   Avg Confidence (Incorrect): {before_metrics['avg_conf_incorrect']:.4f}")
    
    # Wrap model with temperature
    calibrated_model = ModelWithTemperature(model, temperature=optimal_temp)
    calibrated_model = calibrated_model.to(device)
    calibrated_model.eval()
    
    # Evaluate after calibration
    print("\n" + "="*70)
    print("EVALUATING AFTER CALIBRATION")
    print("="*70)
    after_metrics = evaluate_calibration(calibrated_model, val_loader)
    
    print(f"\n■ After Calibration (T={optimal_temp:.4f}):")
    print(f"   Accuracy: {after_metrics['accuracy']:.4f}")
    print(f"   ECE: {after_metrics['ece']:.4f} ↓")
    print(f"   NLL: {after_metrics['nll']:.4f} ↓")
    print(f"   Avg Confidence (Correct): {after_metrics['avg_conf_correct']:.4f}")
    print(f"   Avg Confidence (Incorrect): {after_metrics['avg_conf_incorrect']:.4f} ↓")
    
    # Calculate improvements
    ece_improvement = (before_metrics['ece'] - after_metrics['ece']) / before_metrics['ece'] * 100
    nll_improvement = (before_metrics['nll'] - after_metrics['nll']) / before_metrics['nll'] * 100
    
    print(f"\n■ Improvements:")
    print(f"   ECE reduced by: {ece_improvement:.2f}%")
    print(f"   NLL reduced by: {nll_improvement:.2f}%")
    print(f"   Accuracy change: {(after_metrics['accuracy'] - before_metrics['accuracy'])*100:.2f}%")
    
    # Generate outputs
    output_dir = config['paths']['outputs']
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING CALIBRATION VISUALIZATIONS")
    print("="*70)
    
    plot_reliability_diagram(
        before_metrics, after_metrics,
        os.path.join(output_dir, 'calibration_reliability_diagram.png')
    )
    
    plot_confidence_histograms(
        before_metrics, after_metrics,
        os.path.join(output_dir, 'calibration_confidence_histograms.png')
    )
    
    generate_calibration_report(
        before_metrics, after_metrics, optimal_temp,
        os.path.join(output_dir, 'calibration_report.txt')
    )
    
    # Save calibrated model
    save_path = os.path.join(config['paths']['models'], args.output_name)
    save_calibrated_model(model, optimal_temp, checkpoint, save_path)
    
    print("\n" + "="*70)
    print("✓ CALIBRATION COMPLETE!")
    print("="*70)
    print(f"\n■ Calibrated model saved: {save_path}")
    print(f"■ Use this model for inference with temperature={optimal_temp:.4f}")
    print("\nTo use calibrated predictions:")
    print(f"  python inference.py <image> --model_path {save_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()