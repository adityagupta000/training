import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from tqdm import tqdm

from model import load_model
from data_loader import get_data_loaders

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create output directory
os.makedirs(config['paths']['outputs'], exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_model(model_path, test_loader):
    """
    Comprehensive model evaluation
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70 + "\n")
    
    # Load model
    print(f"■ Loading model from: {model_path}")
    model, checkpoint = load_model(model_path, device)
    model.eval()
    
    # Get predictions
    print("■ Generating predictions...")
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_pred_proba.extend(probabilities.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    return y_true, y_pred, y_pred_proba


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot and save confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{config['paths']['outputs']}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved")


def plot_per_class_metrics(y_true, y_pred, class_names):
    """
    Plot per-class precision, recall, F1-score
    """
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    classes = class_names
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(f"{config['paths']['outputs']}/per_class_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Per-class metrics plot saved")


def plot_roc_curves(y_true, y_pred_proba, class_names):
    """
    Plot ROC curves for each class
    """
    n_classes = len(class_names)
    
    # Binarize labels
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#95a5a6']  
    
    for i, (color, class_name) in enumerate(zip(colors, class_names)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
        
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Multi-class', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{config['paths']['outputs']}/roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ ROC curves saved")


def plot_confidence_distribution(y_pred_proba, y_true, y_pred):
    """
    Plot confidence score distribution for correct vs incorrect predictions
    """
    max_proba = np.max(y_pred_proba, axis=1)
    correct = (y_true == y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(max_proba[correct], bins=30, alpha=0.7, label='Correct', color='#2ecc71')
    ax.hist(max_proba[~correct], bins=30, alpha=0.7, label='Incorrect', color='#e74c3c')
    
    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Confidence Distribution: Correct vs Incorrect Predictions', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{config['paths']['outputs']}/confidence_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confidence distribution plot saved")


def generate_evaluation_report(y_true, y_pred, y_pred_proba, class_names):
    """
    Generate comprehensive text report
    """
    report_path = f"{config['paths']['outputs']}/evaluation_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PLANT HEALTH MONITORING - EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Overall accuracy
        accuracy = np.mean(y_true == y_pred)
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        
        # Classification report
        f.write("="*70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names))
        f.write("\n")
        
        # Confusion matrix
        f.write("="*70 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("="*70 + "\n\n")
        cm = confusion_matrix(y_true, y_pred)
        f.write(f"{'':20}" + "".join([f"{c:>15}" for c in class_names]) + "\n")
        for i, row in enumerate(cm):
            f.write(f"{class_names[i]:20}" + "".join([f"{val:>15}" for val in row]) + "\n")
        f.write("\n")
        
        # Per-class metrics
        f.write("="*70 + "\n")
        f.write("PER-CLASS DETAILED METRICS\n")
        f.write("="*70 + "\n\n")
        
        report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        for class_name in class_names:
            f.write(f"{class_name}:\n")
            f.write(f"  Precision: {report_dict[class_name]['precision']:.4f}\n")
            f.write(f"  Recall: {report_dict[class_name]['recall']:.4f}\n")
            f.write(f"  F1-Score: {report_dict[class_name]['f1-score']:.4f}\n")
            f.write(f"  Support: {report_dict[class_name]['support']:.0f}\n\n")
        
        # Confidence statistics
        max_proba = np.max(y_pred_proba, axis=1)
        correct = (y_true == y_pred)
        
        f.write("="*70 + "\n")
        f.write("CONFIDENCE STATISTICS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Average confidence (correct predictions): {np.mean(max_proba[correct]):.4f}\n")
        f.write(f"Average confidence (incorrect predictions): {np.mean(max_proba[~correct]):.4f}\n")
        f.write(f"Overall average confidence: {np.mean(max_proba):.4f}\n")
    
    print(f"✓ Evaluation report saved: {report_path}")


def main():
    """Main evaluation pipeline"""
    
    # Load test data
    _, _, test_loader = get_data_loaders(batch_size=16)
    
    # Model path
    model_path = f"{config['paths']['models']}/best_model.pth"
    
    if not os.path.exists(model_path):
        model_path = f"{config['paths']['models']}/model_final.pth"
    
    if not os.path.exists(model_path):
        print(f"✗ No trained model found at {model_path}")
        print("  Please train the model first using train.py")
        return
    
    # Evaluate
    y_true, y_pred, y_pred_proba = evaluate_model(model_path, test_loader)
    
    class_names = config['classes']
    
    # Generate visualizations
    print("\n■ Generating evaluation visualizations...")
    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_per_class_metrics(y_true, y_pred, class_names)
    plot_roc_curves(y_true, y_pred_proba, class_names)
    plot_confidence_distribution(y_pred_proba, y_true, y_pred)
    
    # Generate text report
    print("\n■ Generating evaluation report...")
    generate_evaluation_report(y_true, y_pred, y_pred_proba, class_names)
    
    print("\n" + "="*70)
    print("✓ Evaluation completed successfully!")
    print(f"  All results saved to: {config['paths']['outputs']}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()