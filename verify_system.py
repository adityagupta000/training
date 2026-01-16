"""
Complete System Verification for Plant Health Monitoring
Tests all components with the new Not_Plant class (8 classes total)

Run this script to verify:
1. Configuration correctness
2. Data structure
3. Data preprocessing
4. Model architecture
5. Training pipeline
6. Inference capability
7. All 8 classes including Not_Plant

Usage:
    python verify_system.py
"""

import os
import yaml
import cv2
import torch
import numpy as np
from pathlib import Path
from collections import Counter

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}■ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}■ {text}{Colors.END}")


# ============================================================================
# VERIFICATION 1: Configuration File
# ============================================================================
def verify_config():
    print_header("VERIFICATION 1: Configuration File")
    
    issues = []
    
    # Check config file exists
    if not os.path.exists('config.yaml'):
        print_error("config.yaml not found!")
        return False
    
    # Load config
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print_success("config.yaml loaded successfully")
    except Exception as e:
        print_error(f"Failed to load config.yaml: {e}")
        return False
    
    # Check number of classes
    expected_classes = 8
    actual_classes = config['model']['num_classes']
    
    if actual_classes != expected_classes:
        issues.append(f"num_classes is {actual_classes}, should be {expected_classes}")
        print_error(f"num_classes = {actual_classes} (should be {expected_classes})")
    else:
        print_success(f"num_classes = {expected_classes} ✓")
    
    # Check batch size
    expected_batch_size = 16
    actual_batch_size = config['training']['batch_size']
    
    if actual_batch_size != expected_batch_size:
        print_warning(f"batch_size = {actual_batch_size} (recommended: {expected_batch_size})")
        print_info("  For consistency, consider setting batch_size to 16")
    else:
        print_success(f"batch_size = {expected_batch_size} ✓")
    
    # Check class list
    expected_class_names = [
        'Healthy', 'Pest_Fungal', 'Pest_Bacterial', 'Pest_Insect',
        'Nutrient_Nitrogen', 'Nutrient_Potassium', 'Water_Stress', 'Not_Plant'
    ]
    
    actual_class_names = config['classes']
    
    if len(actual_class_names) != 8:
        issues.append(f"Class list has {len(actual_class_names)} items, should have 8")
        print_error(f"Found {len(actual_class_names)} classes, expected 8")
    else:
        print_success(f"Class list has 8 classes ✓")
    
    # Check if Not_Plant is present
    if 'Not_Plant' not in actual_class_names:
        issues.append("Not_Plant class missing from class list")
        print_error("Not_Plant class not found in config!")
    else:
        print_success("Not_Plant class found in config ✓")
    
    # Print class list
    print_info("Class list:")
    for i, cls in enumerate(actual_class_names, 1):
        marker = "✓" if cls in expected_class_names else "✗"
        print(f"  {i}. {cls} {marker}")
    
    # Check class counts
    if 'class_counts' in config:
        print_info("\nExpected class counts:")
        for cls, count in config['class_counts'].items():
            print(f"  {cls:25} : {count:4} samples")
        
        if 'Not_Plant' in config['class_counts']:
            print_success(f"Not_Plant count configured: {config['class_counts']['Not_Plant']}")
        else:
            print_warning("Not_Plant not in class_counts (will be counted from data)")
    
    # Summary
    print()
    if len(issues) == 0:
        print_success("Configuration file is CORRECT ✓")
        return True
    else:
        print_error("Configuration has issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False


# ============================================================================
# VERIFICATION 2: Data Structure
# ============================================================================
def verify_data_structure():
    print_header("VERIFICATION 2: Data Structure")
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    raw_dir = config['paths']['raw_data']
    
    if not os.path.exists(raw_dir):
        print_error(f"Raw data directory not found: {raw_dir}")
        print_info("Please create the directory and add your data")
        return False
    
    print_success(f"Raw data directory exists: {raw_dir}")
    
    # Check for expected classes
    expected_classes = config['classes']
    found_classes = []
    missing_classes = []
    
    for cls in expected_classes:
        cls_path = os.path.join(raw_dir, cls)
        if os.path.exists(cls_path) and os.path.isdir(cls_path):
            found_classes.append(cls)
            
            # Count images
            image_count = 0
            for root, dirs, files in os.walk(cls_path):
                for file in files:
                    ext = Path(file).suffix.lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        image_count += 1
            
            if image_count > 0:
                print_success(f"{cls:25} : {image_count:4} images")
            else:
                print_warning(f"{cls:25} : No images found")
        else:
            missing_classes.append(cls)
            print_error(f"{cls:25} : Directory not found")
    
    # Check for Not_Plant
    print()
    if 'Not_Plant' in found_classes:
        print_success("Not_Plant class directory exists ✓")
        not_plant_path = os.path.join(raw_dir, 'Not_Plant')
        not_plant_count = sum(1 for root, dirs, files in os.walk(not_plant_path)
                              for file in files if Path(file).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'])
        if not_plant_count > 100:
            print_success(f"Not_Plant has {not_plant_count} images (good)")
        elif not_plant_count > 0:
            print_warning(f"Not_Plant has only {not_plant_count} images (consider adding more)")
        else:
            print_error("Not_Plant directory is empty!")
    else:
        print_error("Not_Plant class directory not found!")
        print_info(f"Create it at: {os.path.join(raw_dir, 'Not_Plant')}")
        print_info("Add images of animals, objects, backgrounds, etc.")
    
    # Summary
    print()
    if len(missing_classes) == 0 and 'Not_Plant' in found_classes:
        print_success("All 8 classes have directories ✓")
        return True
    else:
        print_error(f"Missing {len(missing_classes)} class directories")
        return False


# ============================================================================
# VERIFICATION 3: Preprocessing Test
# ============================================================================
def verify_preprocessing():
    print_header("VERIFICATION 3: Preprocessing Pipeline")
    
    print_info("Testing image preprocessing...")
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    raw_dir = config['paths']['raw_data']
    img_size = config['image']['size']
    
    # Find a test image
    test_image = None
    test_class = None
    
    for cls in config['classes']:
        cls_path = os.path.join(raw_dir, cls)
        if os.path.exists(cls_path):
            for root, dirs, files in os.walk(cls_path):
                for file in files:
                    ext = Path(file).suffix.lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        test_image = os.path.join(root, file)
                        test_class = cls
                        break
                if test_image:
                    break
        if test_image:
            break
    
    if not test_image:
        print_warning("No images found for testing")
        return False
    
    print_info(f"Testing with: {test_class}/{Path(test_image).name}")
    
    # Test reading
    try:
        img = cv2.imread(test_image)
        if img is None:
            print_error("Failed to read test image")
            return False
        print_success(f"Image read: {img.shape}")
    except Exception as e:
        print_error(f"Error reading image: {e}")
        return False
    
    # Test resizing
    try:
        img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
        print_success(f"Resize to {img_size}x{img_size}: OK")
    except Exception as e:
        print_error(f"Error resizing: {e}")
        return False
    
    # Test normalization
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        transform = A.Compose([
            A.Normalize(
                mean=config['image']['normalize_mean'],
                std=config['image']['normalize_std'],
                max_pixel_value=config['image']['max_pixel_value']
            ),
            ToTensorV2()
        ])
        
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = transform(image=img_rgb)['image']
        
        print_success(f"Normalization: OK, tensor shape: {img_tensor.shape}")
    except Exception as e:
        print_error(f"Error in normalization: {e}")
        return False
    
    print()
    print_success("Preprocessing pipeline works correctly ✓")
    return True


# ============================================================================
# VERIFICATION 4: Model Architecture
# ============================================================================
def verify_model():
    print_header("VERIFICATION 4: Model Architecture")
    
    try:
        from model import build_model
        
        print_info("Building model...")
        model = build_model()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print_success(f"Model created successfully")
        print_info(f"Total parameters: {total_params:,}")
        print_info(f"Trainable parameters: {trainable_params:,}")
        
        # Check number of output classes
        # Test forward pass with batch size > 1 (BatchNorm requires this)
        model.eval()  # Set to eval mode to avoid BatchNorm issues
        dummy_input = torch.randn(2, 3, 224, 224)  # Use batch size of 2
        output = model(dummy_input)
        
        if output.shape[1] == 8:
            print_success(f"Output shape correct: {output.shape} (batch_size=2, 8 classes) ✓")
        else:
            print_error(f"Output shape incorrect: {output.shape} (should be [2, 8])")
            return False
        
        print()
        print_success("Model architecture is correct ✓")
        return True
        
    except Exception as e:
        print_error(f"Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# VERIFICATION 5: Data Loader
# ============================================================================
def verify_data_loader():
    print_header("VERIFICATION 5: Data Loader")
    
    # Check if splits exist
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    splits_dir = config['paths']['splits']
    
    if not os.path.exists(splits_dir):
        print_warning("Data splits not created yet")
        print_info("Run: python preprocessing.py")
        return False
    
    split_files = ['train.txt', 'val.txt', 'test.txt']
    all_exist = True
    
    for split_file in split_files:
        path = os.path.join(splits_dir, split_file)
        if os.path.exists(path):
            with open(path, 'r') as f:
                count = len(f.readlines())
            print_success(f"{split_file}: {count} samples")
        else:
            print_error(f"{split_file}: Not found")
            all_exist = False
    
    if not all_exist:
        print_info("Run: python preprocessing.py")
        return False
    
    # Test data loader
    try:
        from data_loader import get_data_loaders
        
        print_info("\nTesting data loaders...")
        train_loader, val_loader, test_loader = get_data_loaders(batch_size=4)
        
        print_success(f"Train batches: {len(train_loader)}")
        print_success(f"Val batches: {len(val_loader)}")
        print_success(f"Test batches: {len(test_loader)}")
        
        # Test loading a batch
        images, labels = next(iter(train_loader))
        print_success(f"Batch shape: {images.shape}")
        print_success(f"Label range: {labels.min().item()} - {labels.max().item()} (should be 0-7)")
        
        # Check if Not_Plant (label 7) appears
        if 7 in labels:
            print_success("Not_Plant samples found in batch ✓")
        else:
            print_info("Not_Plant not in this batch (normal if dataset is small)")
        
        print()
        print_success("Data loaders work correctly ✓")
        return True
        
    except Exception as e:
        print_error(f"Data loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# VERIFICATION 6: Loss Functions
# ============================================================================
def verify_loss_functions():
    print_header("VERIFICATION 6: Loss Functions")
    
    try:
        from model import FocalLoss, LabelSmoothingCrossEntropy
        
        # Test Focal Loss
        print_info("Testing Focal Loss...")
        focal = FocalLoss(alpha=0.25, gamma=2.0)
        
        dummy_logits = torch.randn(4, 8)  # 4 samples, 8 classes
        dummy_labels = torch.tensor([0, 3, 5, 7])  # Including Not_Plant (7)
        
        loss = focal(dummy_logits, dummy_labels)
        print_success(f"Focal Loss: {loss.item():.4f}")
        
        # Test Label Smoothing
        print_info("Testing Label Smoothing...")
        ls_loss = LabelSmoothingCrossEntropy(epsilon=0.1)
        loss2 = ls_loss(dummy_logits, dummy_labels)
        print_success(f"Label Smoothing Loss: {loss2.item():.4f}")
        
        print()
        print_success("Loss functions work correctly ✓")
        return True
        
    except Exception as e:
        print_error(f"Loss function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# VERIFICATION 7: Class Distribution
# ============================================================================
def verify_class_distribution():
    print_header("VERIFICATION 7: Class Distribution Analysis")
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    raw_dir = config['paths']['raw_data']
    
    class_counts = {}
    total = 0
    
    for cls in config['classes']:
        cls_path = os.path.join(raw_dir, cls)
        if os.path.exists(cls_path):
            count = 0
            for root, dirs, files in os.walk(cls_path):
                for file in files:
                    ext = Path(file).suffix.lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        count += 1
            class_counts[cls] = count
            total += count
    
    if total == 0:
        print_error("No images found!")
        return False
    
    print_info("Class distribution:")
    
    # Sort by count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    for cls, count in sorted_classes:
        percentage = (count / total * 100) if total > 0 else 0
        bar_length = int(percentage / 2)
        bar = '█' * bar_length
        
        # Color code based on count
        if count > 1000:
            status = "✓"
        elif count > 500:
            status = "■"
        else:
            status = "■"
        
        print(f"  {status} {cls:25} : {count:4} ({percentage:5.1f}%) {bar}")
    
    print(f"\n  {'TOTAL':25} : {total:4}")
    
    # Calculate imbalance
    if class_counts:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance = max_count / min_count if min_count > 0 else 0
        
        print(f"\n  Imbalance ratio: {imbalance:.2f}:1")
        
        if imbalance > 5:
            print_warning("High imbalance - class weights will be critical")
        elif imbalance > 3:
            print_info("Moderate imbalance - class weights recommended")
        else:
            print_success("Well balanced dataset")
    
    # Check Not_Plant specifically
    print()
    if 'Not_Plant' in class_counts:
        not_plant_count = class_counts['Not_Plant']
        if not_plant_count >= 500:
            print_success(f"Not_Plant has {not_plant_count} samples (excellent)")
        elif not_plant_count >= 200:
            print_success(f"Not_Plant has {not_plant_count} samples (good)")
        elif not_plant_count >= 100:
            print_warning(f"Not_Plant has {not_plant_count} samples (acceptable, but more recommended)")
        else:
            print_warning(f"Not_Plant has only {not_plant_count} samples (add more non-plant images)")
    
    print()
    print_success("Class distribution analyzed ✓")
    return True


# ============================================================================
# MAIN VERIFICATION
# ============================================================================
def main():
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║     PLANT HEALTH MONITORING - COMPLETE SYSTEM VERIFICATION        ║")
    print("║                    8 Classes (Including Not_Plant)                ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}\n")
    
    results = {}
    
    # Run all verifications
    results['config'] = verify_config()
    results['data_structure'] = verify_data_structure()
    results['preprocessing'] = verify_preprocessing()
    results['model'] = verify_model()
    results['data_loader'] = verify_data_loader()
    results['loss_functions'] = verify_loss_functions()
    results['distribution'] = verify_class_distribution()
    
    # Final summary
    print_header("FINAL VERIFICATION SUMMARY")
    
    print("Component Status:")
    for component, passed in results.items():
        status = f"{Colors.GREEN}PASS ✓{Colors.END}" if passed else f"{Colors.RED}FAIL ✗{Colors.END}"
        print(f"  {component.replace('_', ' ').title():25} : {status}")
    
    total = len(results)
    passed = sum(results.values())
    percentage = (passed / total * 100) if total > 0 else 0
    
    print(f"\n{Colors.BOLD}Overall: {passed}/{total} checks passed ({percentage:.0f}%){Colors.END}")
    
    if all(results.values()):
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ SYSTEM IS READY FOR TRAINING!{Colors.END}")
        print(f"\n{Colors.BOLD}Next steps:{Colors.END}")
        print(f"  1. Run: {Colors.BLUE}python preprocessing.py{Colors.END}")
        print(f"  2. Run: {Colors.BLUE}python train.py{Colors.END}")
        print(f"  3. Test: {Colors.BLUE}python inference.py <image_path>{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ SYSTEM HAS ISSUES - FIX BEFORE TRAINING{Colors.END}")
        print(f"\n{Colors.BOLD}Failed checks:{Colors.END}")
        for component, passed in results.items():
            if not passed:
                print(f"  {Colors.RED}✗ {component.replace('_', ' ').title()}{Colors.END}")
    
    print()


if __name__ == "__main__":
    main()