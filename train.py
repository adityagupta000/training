import os
import yaml
import argparse
import numpy as np
import pandas as pd  # ADD THIS IMPORT
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix

from model import (build_model, print_model_summary, save_model,
                   FocalLoss, LabelSmoothingCrossEntropy)
from data_loader import (get_data_loaders, get_balanced_class_weights, mixup_data)

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

os.makedirs(config['paths']['models'], exist_ok=True)
os.makedirs(config['paths']['logs'], exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_loss_function(class_weights=None):
    """Return calibrated loss function"""
    use_focal = config['training'].get('use_focal_loss', True)
    use_label_smoothing = config['training'].get('label_smoothing', 0.0) > 0
    
    if use_focal:
        focal_gamma = config['training'].get('focal_gamma', 2.5)
        focal_alpha = config['training'].get('focal_alpha', 0.25)
        
        criterion = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            weight=class_weights,
            reduction='mean'
        )
        print(f"\n✓ Focal Loss: gamma={focal_gamma}, alpha={focal_alpha}")
        print(f"  Higher gamma (2.5) for your 5.56:1 imbalance")
    
    elif use_label_smoothing:
        epsilon = config['training']['label_smoothing']
        criterion = LabelSmoothingCrossEntropy(
            epsilon=epsilon,
            weight=class_weights,
            reduction='mean'
        )
        print(f"\n✓ Label Smoothing CE: epsilon={epsilon}")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("\n✓ Weighted Cross-Entropy")
    
    return criterion


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss computation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(model, loader, criterion, optimizer, scaler, use_amp=True, use_mixup=False):
    """Training with optional mixup"""
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Training')
    
    for batch_data in pbar:
        images, labels = batch_data
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Apply mixup
        if use_mixup and np.random.rand() < 0.5:
            mixed_images, labels_a, labels_b, lam = mixup_data(
                images, labels,
                alpha=config['training'].get('mixup_alpha', 0.2)
            )
            
            # Forward with mixup
            if use_amp:
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    outputs = model(mixed_images)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(mixed_images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            # For metrics, use original labels
            _, predicted = torch.max(outputs, 1)
            acc = (lam * (predicted == labels_a).float() + 
                   (1 - lam) * (predicted == labels_b).float()).mean()
        else:
            # Standard forward
            if use_amp:
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == labels).float().mean()
        
        # Backward
        optimizer.zero_grad()
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                         config['training']['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                         config['training']['gradient_clip'])
            optimizer.step()
        
        # Track
        losses.update(loss.item(), images.size(0))
        accuracies.update(acc.item(), images.size(0))
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.4f}'
        })
    
    # Macro metrics
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    return losses.avg, accuracies.avg, macro_f1, balanced_acc


def validate(model, loader, criterion, use_amp=True):
    """Comprehensive validation"""
    model.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if use_amp:
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == labels).float().mean()
            
            losses.update(loss.item(), images.size(0))
            accuracies.update(acc.item(), images.size(0))
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accuracies.avg:.4f}'
            })
    
    # Metrics
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    return (losses.avg, accuracies.avg, macro_f1, balanced_acc,
            per_class_f1, all_labels, all_preds, all_probs)


def print_per_class_metrics(per_class_f1, classes, class_counts=None):
    """Print detailed per-class metrics with context"""
    print("\n  Per-Class F1 Scores:")
    
    for i, (cls, f1) in enumerate(zip(classes, per_class_f1)):
        # Status indicator
        if f1 > 0.65:
            status = "✓"
        elif f1 > 0.45:
            status = "■"
        elif f1 > 0.25:
            status = "■"
        else:
            status = "✗"
        
        # Count info
        count_str = ""
        if class_counts:
            count = class_counts.get(cls, 0)
            if count < 700:
                count_str = f"  [■ {count} samples]"
            elif count < 1500:
                count_str = f"  [■ {count} samples]"
            else:
                count_str = f"  [■ {count} samples]"
        
        print(f"  {status} {cls:25} : {f1:.3f}{count_str}")


def check_class_confusion(labels, preds, classes):
    """Analyze confusion between specific classes"""
    cm = confusion_matrix(labels, preds)
    
    # Check if Healthy is being confused with Nutrients
    healthy_idx = classes.index('Healthy')
    n_nitrogen_idx = classes.index('Nutrient_Nitrogen')
    n_potassium_idx = classes.index('Nutrient_Potassium')
    
    healthy_total = cm[healthy_idx].sum()
    if healthy_total > 0:
        healthy_to_nitrogen = cm[healthy_idx, n_nitrogen_idx] / healthy_total
        healthy_to_potassium = cm[healthy_idx, n_potassium_idx] / healthy_total
        
        if healthy_to_nitrogen > 0.15 or healthy_to_potassium > 0.15:
            print("\n  ■■ WARNING: Healthy samples being misclassified as Nutrients!")
            print(f"    Healthy → Nitrogen: {healthy_to_nitrogen*100:.1f}%")
            print(f"    Healthy → Potassium: {healthy_to_potassium*100:.1f}%")
            return True
    
    return False


# ADD THIS NEW FUNCTION
def save_history_to_csv(history, phase_name):
    """Save training history to CSV file"""
    logs_dir = config['paths']['logs']
    csv_path = os.path.join(logs_dir, f'training_{phase_name}.csv')
    
    # Convert history list of dicts to DataFrame
    df = pd.DataFrame(history)
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Training logs saved: {csv_path}")
    
    return csv_path


def train_phase1(model, train_loader, val_loader, criterion, optimizer,
                 scheduler, scaler, max_epochs, patience, use_amp=True, use_mixup=False):
    """Phase 1 with calibrated metrics"""
    print("\n" + "="*70)
    print("PHASE 1: Training Classification Head")
    print("="*70)
    print("■ Optimized for YOUR distribution (5.56:1 imbalance)")
    print("■ Capped weights prevent over-prediction")
    print("="*70 + "\n")
    
    best_macro_f1 = 0.0
    epochs_no_improve = 0
    history = []
    
    # Class counts for reference
    class_counts = config.get('class_counts', {})
    
    for epoch in range(max_epochs):
        print(f"\nEpoch [{epoch+1}/{max_epochs}]")
        print("-" * 70)
        
        # Train
        train_loss, train_acc, train_f1, train_bal_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, use_amp, use_mixup
        )
        
        # Validate
        (val_loss, val_acc, val_f1, val_bal_acc, per_class_f1,
         val_labels, val_preds, _) = validate(model, val_loader, criterion, use_amp)
        
        scheduler.step(val_f1)
        
        # Print
        print(f"\n  Train: Loss={train_loss:.4f} | Acc={train_acc*100:.2f}% | "
              f"F1={train_f1:.3f} | Bal-Acc={train_bal_acc:.3f}")
        print(f"  Val:   Loss={val_loss:.4f} | Acc={val_acc*100:.2f}% | "
              f"F1={val_f1:.3f} | Bal-Acc={val_bal_acc:.3f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        print_per_class_metrics(per_class_f1, config['classes'], class_counts)
        
        # Check for problematic confusion
        confusion_issue = check_class_confusion(val_labels, val_preds, config['classes'])
        
        # History - ADD PROPER STRUCTURE FOR CSV
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'train_bal_acc': train_bal_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_bal_acc': val_bal_acc,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Check improvement
        if val_f1 > best_macro_f1:
            improvement = val_f1 - best_macro_f1
            best_macro_f1 = val_f1
            epochs_no_improve = 0
            print(f"\n  ✓ NEW BEST Macro-F1: {best_macro_f1:.3f} (+{improvement:.3f})")
            
            save_path = os.path.join(config['paths']['models'], 'best_model_phase1.pth')
            save_model(model, optimizer, epoch, best_macro_f1, save_path, 'macro_f1')
        else:
            epochs_no_improve += 1
            print(f"\n  ■ No improvement: {epochs_no_improve}/{patience} epochs")
        
        # Phase transition
        if epochs_no_improve >= patience:
            print(f"\n{'='*70}")
            print(f"■ PLATEAU → Transitioning to Phase 2")
            print(f"{'='*70}\n")
            break
    
    print(f"\n■ Phase 1 Complete: {epoch+1} epochs")
    print(f"  Best Macro-F1: {best_macro_f1:.3f}")
    
    # SAVE PHASE 1 HISTORY TO CSV
    save_history_to_csv(history, 'phase1')
    
    return history, best_macro_f1


def train_phase2(model, train_loader, val_loader, criterion, optimizer,
                 scheduler, scaler, max_epochs, patience, use_amp=True, use_mixup=False):
    """Phase 2 with very low LR"""
    print("\n" + "="*70)
    print("PHASE 2: Fine-Tuning Entire Model")
    print("="*70)
    print("■ VERY LOW LR to preserve minority class features")
    print("="*70 + "\n")
    
    best_macro_f1 = 0.0
    epochs_no_improve = 0
    history = []
    
    class_counts = config.get('class_counts', {})
    
    for epoch in range(max_epochs):
        print(f"\nEpoch [{epoch+1}/{max_epochs}]")
        print("-" * 70)
        
        train_loss, train_acc, train_f1, train_bal_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, use_amp, use_mixup
        )
        
        (val_loss, val_acc, val_f1, val_bal_acc, per_class_f1,
         val_labels, val_preds, _) = validate(model, val_loader, criterion, use_amp)
        
        scheduler.step(val_f1)
        
        print(f"\n  Train: Loss={train_loss:.4f} | Acc={train_acc*100:.2f}% | "
              f"F1={train_f1:.3f} | Bal-Acc={train_bal_acc:.3f}")
        print(f"  Val:   Loss={val_loss:.4f} | Acc={val_acc*100:.2f}% | "
              f"F1={val_f1:.3f} | Bal-Acc={val_bal_acc:.3f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        print_per_class_metrics(per_class_f1, config['classes'], class_counts)
        confusion_issue = check_class_confusion(val_labels, val_preds, config['classes'])
        
        # History - ADD PROPER STRUCTURE FOR CSV
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'train_bal_acc': train_bal_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_bal_acc': val_bal_acc,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        if val_f1 > best_macro_f1:
            improvement = val_f1 - best_macro_f1
            best_macro_f1 = val_f1
            epochs_no_improve = 0
            print(f"\n  ✓ NEW BEST Macro-F1: {best_macro_f1:.3f} (+{improvement:.3f})")
            
            save_path = os.path.join(config['paths']['models'], 'best_model.pth')
            save_model(model, optimizer, epoch, best_macro_f1, save_path, 'macro_f1')
        else:
            epochs_no_improve += 1
            print(f"\n  ■ No improvement: {epochs_no_improve}/{patience} epochs")
        
        if epochs_no_improve >= patience:
            print(f"\n{'='*70}")
            print(f"■ EARLY STOPPING")
            print(f"{'='*70}\n")
            break
    
    print(f"\n■ Phase 2 Complete: {epoch+1} epochs")
    print(f"  Best Macro-F1: {best_macro_f1:.3f}")
    
    # SAVE PHASE 2 HISTORY TO CSV
    save_history_to_csv(history, 'phase2')
    
    return history, best_macro_f1


def main(args):
    """Calibrated training pipeline"""
    print("\n" + "="*70)
    print("CALIBRATED TRAINING - Plant Health Monitoring")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print("\n■ OPTIMIZED FOR YOUR DISTRIBUTION:")
    print("  • Pest_Bacterial: 2780 (MAJORITY)")
    print("  • Healthy: 1836")
    print("  • Pest_Fungal: 1720")
    print("  • Pest_Insect: 991")
    print("  • Nutrient_Nitrogen: 500 (MINORITY)")
    print("  • Nutrient_Potassium: 500 (MINORITY)")
    print("  • Water_Stress: 568 (MINORITY)")
    print("  • Not_Plant: 409 (MINORITY)")
    print(f"\n  Imbalance Ratio: 6.80:1")
    print("="*70)
    print("\n■ CALIBRATIONS APPLIED:")
    print("  ✓ Capped class weights (max 5.0)")
    print("  ✓ Focal Loss gamma=2.5 (for 5.56:1)")
    print("  ✓ Moderate oversampling (not aggressive)")
    print("  ✓ Mixup augmentation for boundaries")
    print("  ✓ Confusion monitoring (Healthy vs Nutrients)")
    print("="*70 + "\n")
    
    # Seeds
    torch.manual_seed(config['data']['random_seed'])
    np.random.seed(config['data']['random_seed'])
    
    # Model
    model = build_model(
        model_name=args.model_name,
        num_classes=config['model']['num_classes'],
        pretrained=not args.no_pretrained
    )
    model = model.to(device)
    print_model_summary(model)
    
    # Data
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=config['training']['batch_size'],
        use_weighted_sampler=not args.no_weighted_sampler
    )
    
    # Class weights
    class_weights = None
    if not args.no_class_weights:
        splits_file = os.path.join(config['paths']['splits'], 'train.txt')
        method = config['training'].get('class_weight_method', 'balanced')
        power = config['training'].get('class_weight_power', 1.0)
        max_cap = config['training'].get('max_weight_cap', 5.0)
        
        class_weights = get_balanced_class_weights(
            splits_file,
            method=method,
            power=power,
            max_cap=max_cap
        )
        class_weights = class_weights.to(device)
    
    # Loss
    criterion = get_loss_function(class_weights)
    
    # Mixed precision
    use_amp = config['training']['use_mixed_precision'] and not args.no_mixed_precision
    scaler = GradScaler(enabled=use_amp)
    
    # Mixup
    use_mixup = config['training'].get('mixup_alpha', 0.0) > 0
    if use_mixup:
        print(f"\n✓ Mixup enabled (alpha={config['training']['mixup_alpha']})")
    
    # ==================== PHASE 1 ====================
    print("\n" + "="*70)
    print("PHASE 1 START")
    print("="*70)
    
    optimizer_p1 = optim.AdamW(
        model.classifier.parameters(),
        lr=config['training']['initial_lr'],
        weight_decay=1e-4
    )
    
    scheduler_p1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p1,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=config['training']['min_lr'],
    )
    
    history1, best_f1_p1 = train_phase1(
        model, train_loader, val_loader, criterion,
        optimizer_p1, scheduler_p1, scaler,
        max_epochs=args.max_epochs_phase1,
        patience=config['training']['phase1_transition_patience'],
        use_amp=use_amp,
        use_mixup=use_mixup
    )
    
    # ==================== PHASE 2 ====================
    print("\n" + "="*70)
    print("PHASE 2 START")
    print("="*70)
    
    model.unfreeze_base()
    
    # CRITICAL: Very low LR
    lr_phase2 = config['training']['initial_lr'] / 25
    
    optimizer_p2 = optim.AdamW(
        model.parameters(),
        lr=lr_phase2,
        weight_decay=1e-4
    )
    
    print(f"\n✓ Phase 2 LR: {lr_phase2:.7f} (1/25 of Phase 1)")
    print("  → Extremely gentle fine-tuning\n")
    
    scheduler_p2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p2,
        mode='max',
        factor=0.5,
        patience=10,
        min_lr=config['training']['min_lr'],
    )
    
    history2, best_f1_p2 = train_phase2(
        model, train_loader, val_loader, criterion,
        optimizer_p2, scheduler_p2, scaler,
        max_epochs=args.max_epochs_phase2,
        patience=config['training']['patience'],
        use_amp=use_amp,
        use_mixup=use_mixup
    )
    
    # ==================== FINAL TEST ====================
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION")
    print("="*70 + "\n")
    
    best_path = os.path.join(config['paths']['models'], 'best_model.pth')
    checkpoint = torch.load(best_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    (test_loss, test_acc, test_f1, test_bal_acc, test_per_class_f1,
     test_labels, test_preds, _) = validate(model, test_loader, criterion, use_amp)
    
    # Analysis
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\n■ Test Metrics:")
    print(f"  Accuracy: {test_acc*100:.2f}%")
    print(f"  Macro-F1: {test_f1:.3f} ■")
    print(f"  Balanced Accuracy: {test_bal_acc:.3f}")
    
    print("\n■ Per-Class Performance:")
    print_per_class_metrics(test_per_class_f1, config['classes'], config.get('class_counts'))
    
    # Check specific issues
    check_class_confusion(test_labels, test_preds, config['classes'])
    
    # Minority analysis
    minority_indices = [4, 5, 6, 7]  # Nitrogen, Potassium, Water_Stress, Not_Plant
    minority_f1 = np.mean([test_per_class_f1[i] for i in minority_indices])
    
    print(f"\n■ Minority Classes Avg F1: {minority_f1:.3f}")
    
    if minority_f1 < 0.35:
        print("  ■■ Still struggling - consider:")
        print("    • Collect more real minority samples")
        print("    • Increase focal_gamma to 3.0")
        print("    • Use aggressive sampling strategy")
    elif minority_f1 < 0.50:
        print("  ■ Improving but suboptimal")
    else:
        print("  ✓ Good minority performance!")
    
    # Save summary
    log_path = os.path.join(config['paths']['logs'], 'calibrated_results.txt')
    with open(log_path, 'w') as f:
        f.write("CALIBRATED TRAINING RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total Epochs: {len(history1) + len(history2)}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Macro-F1: {test_f1:.4f}\n")
        f.write(f"Minority Avg F1: {minority_f1:.4f}\n\n")
        f.write("Per-Class F1:\n")
        for cls, f1 in zip(config['classes'], test_per_class_f1):
            f.write(f"  {cls}: {f1:.4f}\n")
    
    print(f"\n✓ Results saved: {log_path}")
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calibrated Training')
    
    parser.add_argument('--model_name', type=str,
                       default=None,
                       help='Model name (defaults to config.yaml)')
    
    parser.add_argument('--max_epochs_phase1', type=int, default=60)
    parser.add_argument('--max_epochs_phase2', type=int, default=140)
    
    parser.add_argument('--no_pretrained', action='store_true')
    parser.add_argument('--no_mixed_precision', action='store_true')
    parser.add_argument('--no_class_weights', action='store_true')
    parser.add_argument('--no_weighted_sampler', action='store_true')
    
    args = parser.parse_args()
    
    main(args)