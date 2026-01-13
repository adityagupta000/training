import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)


def load_training_logs():
    """
    Load training logs from CSV files (if available)
    """
    logs_dir = config['paths']['logs']
    
    phase1_path = os.path.join(logs_dir, 'training_phase1.csv')
    phase2_path = os.path.join(logs_dir, 'training_phase2.csv')
    
    phase1_df = None
    phase2_df = None
    
    if os.path.exists(phase1_path):
        phase1_df = pd.read_csv(phase1_path)
        print(f"✓ Loaded Phase 1 logs: {len(phase1_df)} epochs")
    else:
        print(f"■ Warning: Phase 1 logs not found at {phase1_path}")
    
    if os.path.exists(phase2_path):
        phase2_df = pd.read_csv(phase2_path)
        print(f"✓ Loaded Phase 2 logs: {len(phase2_df)} epochs")
    else:
        print(f"■ Warning: Phase 2 logs not found at {phase2_path}")
    
    return phase1_df, phase2_df


def plot_combined_metrics(phase1_df=None, phase2_df=None):
    """
    Plot training and validation metrics across both phases
    """
    output_dir = config['paths']['outputs']
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training History - Two-Phase Training', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Combine data
    combined_train_loss = []
    combined_val_loss = []
    combined_train_acc = []
    combined_val_acc = []
    phase_labels = []
    
    phase1_epochs = 0
    if phase1_df is not None:
        phase1_epochs = len(phase1_df)
        combined_train_loss.extend(phase1_df['train_loss'].tolist())
        combined_val_loss.extend(phase1_df['val_loss'].tolist())
        combined_train_acc.extend(phase1_df['train_acc'].tolist())
        combined_val_acc.extend(phase1_df['val_acc'].tolist())
        phase_labels.extend(['Phase 1'] * phase1_epochs)
    
    if phase2_df is not None:
        combined_train_loss.extend(phase2_df['train_loss'].tolist())
        combined_val_loss.extend(phase2_df['val_loss'].tolist())
        combined_train_acc.extend(phase2_df['train_acc'].tolist())
        combined_val_acc.extend(phase2_df['val_acc'].tolist())
        phase_labels.extend(['Phase 2'] * len(phase2_df))
    
    epochs = range(1, len(combined_train_loss) + 1)
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, combined_train_loss, 'b-', label='Training Loss', linewidth=2)
    if phase1_epochs > 0:
        ax1.axvline(x=phase1_epochs, color='red', linestyle='--', 
                   label='Phase Transition', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax2 = axes[0, 1]
    ax2.plot(epochs, combined_val_loss, 'g-', label='Validation Loss', linewidth=2)
    if phase1_epochs > 0:
        ax2.axvline(x=phase1_epochs, color='red', linestyle='--', 
                   label='Phase Transition', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Accuracy
    ax3 = axes[1, 0]
    ax3.plot(epochs, [acc * 100 for acc in combined_train_acc], 
            'b-', label='Training Accuracy', linewidth=2)
    if phase1_epochs > 0:
        ax3.axvline(x=phase1_epochs, color='red', linestyle='--', 
                   label='Phase Transition', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 105])
    
    # Plot 4: Validation Accuracy
    ax4 = axes[1, 1]
    ax4.plot(epochs, [acc * 100 for acc in combined_val_acc], 
            'g-', label='Validation Accuracy', linewidth=2)
    if phase1_epochs > 0:
        ax4.axvline(x=phase1_epochs, color='red', linestyle='--', 
                   label='Phase Transition', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Accuracy (%)', fontsize=12)
    ax4.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 105])
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'training_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training visualization saved: {output_path}")


def plot_comparison_metrics(phase1_df=None, phase2_df=None):
    """
    Plot side-by-side comparison of Phase 1 vs Phase 2
    """
    if phase1_df is None or phase2_df is None:
        print("■ Cannot create comparison plot - missing phase data")
        return
    
    output_dir = config['paths']['outputs']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Phase Comparison: Classification Head vs Full Fine-Tuning', 
                 fontsize=16, fontweight='bold')
    
    # Phase 1 metrics
    phase1_final_train = phase1_df['train_acc'].iloc[-1] * 100
    phase1_final_val = phase1_df['val_acc'].iloc[-1] * 100
    phase1_best_val = phase1_df['val_acc'].max() * 100
    
    # Phase 2 metrics
    phase2_final_train = phase2_df['train_acc'].iloc[-1] * 100
    phase2_final_val = phase2_df['val_acc'].iloc[-1] * 100
    phase2_best_val = phase2_df['val_acc'].max() * 100
    
    # Bar chart 1: Final Accuracies
    ax1 = axes[0]
    x = ['Phase 1', 'Phase 2']
    train_accs = [phase1_final_train, phase2_final_train]
    val_accs = [phase1_final_val, phase2_final_val]
    
    x_pos = range(len(x))
    width = 0.35
    
    ax1.bar([p - width/2 for p in x_pos], train_accs, width, 
           label='Train Accuracy', color='#3498db')
    ax1.bar([p + width/2 for p in x_pos], val_accs, width, 
           label='Val Accuracy', color='#2ecc71')
    
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Final Accuracies', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # Add value labels on bars
    for i, (train, val) in enumerate(zip(train_accs, val_accs)):
        ax1.text(i - width/2, train + 1, f'{train:.1f}%', 
                ha='center', fontweight='bold')
        ax1.text(i + width/2, val + 1, f'{val:.1f}%', 
                ha='center', fontweight='bold')
    
    # Bar chart 2: Best Validation Accuracies
    ax2 = axes[1]
    best_vals = [phase1_best_val, phase2_best_val]
    
    bars = ax2.bar(x, best_vals, color=['#3498db', '#2ecc71'])
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Best Validation Accuracies', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 105])
    
    # Add value labels on bars
    for bar, val in zip(bars, best_vals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'phase_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Phase comparison saved: {output_path}")


def plot_learning_rate_schedule(phase1_df=None, phase2_df=None):
    """
    Plot learning rate changes across training
    """
    if phase1_df is None and phase2_df is None:
        print("■ Cannot create LR plot - no phase data available")
        return
    
    output_dir = config['paths']['outputs']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    lrs = []
    epochs = []
    phase_labels = []
    
    current_epoch = 0
    
    if phase1_df is not None and 'lr' in phase1_df.columns:
        phase1_epochs = len(phase1_df)
        lrs.extend(phase1_df['lr'].tolist())
        epochs.extend(range(current_epoch + 1, current_epoch + phase1_epochs + 1))
        phase_labels.extend(['Phase 1'] * phase1_epochs)
        current_epoch += phase1_epochs
    
    if phase2_df is not None and 'lr' in phase2_df.columns:
        phase2_epochs = len(phase2_df)
        lrs.extend(phase2_df['lr'].tolist())
        epochs.extend(range(current_epoch + 1, current_epoch + phase2_epochs + 1))
        phase_labels.extend(['Phase 2'] * phase2_epochs)
    
    if len(lrs) == 0:
        print("■ No learning rate data found in logs")
        return
    
    ax.plot(epochs, lrs, 'b-', linewidth=2, marker='o', markersize=4)
    
    if phase1_df is not None:
        ax.axvline(x=len(phase1_df), color='red', linestyle='--', 
                  label='Phase Transition', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'learning_rate_schedule.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Learning rate schedule saved: {output_path}")


def main():
    """Main visualization pipeline"""
    
    print("\n" + "="*70)
    print("TRAINING VISUALIZATION")
    print("="*70 + "\n")
    
    # Load logs
    phase1_df, phase2_df = load_training_logs()
    
    if phase1_df is None and phase2_df is None:
        print("\n✗ No training logs found!")
        print("  Please train the model first using train.py")
        return
    
    # Generate visualizations
    print("\n■ Generating visualizations...")
    
    plot_combined_metrics(phase1_df, phase2_df)
    plot_comparison_metrics(phase1_df, phase2_df)
    plot_learning_rate_schedule(phase1_df, phase2_df)
    
    print("\n" + "="*70)
    print("✓ Visualization completed successfully!")
    print(f"  All plots saved to: {config['paths']['outputs']}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()