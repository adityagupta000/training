import os
import cv2
import yaml
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

IMG_SIZE = config['image']['size']
TRAIN_RATIO = config['data']['train_split']
VAL_RATIO = config['data']['val_split']
RANDOM_SEED = config['data']['random_seed']

RAW_DIR = config['paths']['raw_data']
PROCESSED_DIR = config['paths']['processed_data']
SPLITS_DIR = config['paths']['splits']


def collect_images_from_nested_dirs(class_dir):
    """
    Recursively collect all image files from nested directories
    Handles subdirectories like Pest/Bacterial/, Pest/Fungal/, etc.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    image_files = []
    
    for root, dirs, files in os.walk(class_dir):
        for file in files:
            if Path(file).suffix in image_extensions:
                full_path = os.path.join(root, file)
                image_files.append(full_path)
    
    return image_files


def preprocess_images():
    """
    Preprocess raw images: resize and normalize
    Handles nested directory structure
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print("IMAGE PREPROCESSING PIPELINE")
    print("="*70 + "\n")
    
    total_images = 0
    class_counts = {}
    
    for cls in os.listdir(RAW_DIR):
        raw_cls_path = os.path.join(RAW_DIR, cls)
        
        if not os.path.isdir(raw_cls_path):
            continue
        
        print(f"■ Processing class: {cls}")
        
        # Create output directory
        proc_cls_path = os.path.join(PROCESSED_DIR, cls)
        os.makedirs(proc_cls_path, exist_ok=True)
        
        # Collect all images (including from subdirectories)
        image_files = collect_images_from_nested_dirs(raw_cls_path)
        
        if len(image_files) == 0:
            print(f"  ■ Warning: No images found in {cls}")
            continue
        
        successful = 0
        failed = 0
        
        for img_path in tqdm(image_files, desc=f"  Processing"):
            try:
                # Read image
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"  ✗ Could not read: {os.path.basename(img_path)}")
                    failed += 1
                    continue
                
                # Check if image is too small
                if img.shape[0] < 50 or img.shape[1] < 50:
                    print(f"  ✗ Image too small: {os.path.basename(img_path)}")
                    failed += 1
                    continue
                
                # Resize with high-quality interpolation
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), 
                                interpolation=cv2.INTER_LANCZOS4)
                
                # Generate unique filename (avoid conflicts from subdirs)
                original_name = Path(img_path).stem
                extension = Path(img_path).suffix
                
                # Keep original filename but ensure .jpg extension
                save_name = f"{original_name}.jpg"
                save_path = os.path.join(proc_cls_path, save_name)
                
                # If filename exists, add counter
                counter = 1
                while os.path.exists(save_path):
                    save_name = f"{original_name}_{counter}.jpg"
                    save_path = os.path.join(proc_cls_path, save_name)
                    counter += 1
                
                # Save with high quality
                cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                successful += 1
                
            except Exception as e:
                print(f"  ✗ Error processing {os.path.basename(img_path)}: {e}")
                failed += 1
                continue
        
        class_counts[cls] = successful
        total_images += successful
        
        print(f"  ✓ Processed: {successful} images")
        if failed > 0:
            print(f"  ✗ Failed: {failed} images")
    
    print("\n" + "="*70)
    print("PREPROCESSING SUMMARY")
    print("="*70)
    for cls, count in class_counts.items():
        print(f"  {cls:25} : {count:4} images")
    print(f"  {'TOTAL':25} : {total_images:4} images")
    print("="*70 + "\n")
    
    # Check for class imbalance
    if class_counts:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else 0
        
        if imbalance_ratio > 3:
            print("■ WARNING: Significant class imbalance detected!")
            print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
            print("  Recommendation: Use class weights during training")
            print("  or consider oversampling minority classes.\n")
        
        if min_count < 100:
            print("■ WARNING: Some classes have very few samples!")
            print(f"  Minimum samples: {min_count}")
            print("  Recommendation: Collect more data or use heavy augmentation.\n")
    
    print("✓ Image preprocessing completed\n")


def create_splits():
    """
    Create train/val/test splits with stratification
    Handles class imbalance by ensuring minimum samples in each split
    """
    os.makedirs(SPLITS_DIR, exist_ok=True)
    
    random.seed(RANDOM_SEED)
    
    print("="*70)
    print("CREATING DATA SPLITS")
    print("="*70 + "\n")
    
    train_f = open(os.path.join(SPLITS_DIR, 'train.txt'), 'w')
    val_f = open(os.path.join(SPLITS_DIR, 'val.txt'), 'w')
    test_f = open(os.path.join(SPLITS_DIR, 'test.txt'), 'w')
    
    total_counts = {'train': 0, 'val': 0, 'test': 0}
    class_details = {}
    
    for cls in os.listdir(PROCESSED_DIR):
        cls_path = os.path.join(PROCESSED_DIR, cls)
        
        if not os.path.isdir(cls_path):
            continue
        
        images = [f for f in os.listdir(cls_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        random.shuffle(images)
        
        n = len(images)
        
        if n == 0:
            print(f"■ Warning: No images found for class {cls}")
            continue
        
        # Ensure minimum samples for validation and test
        min_val_test = 2  # At least 2 samples for val and test
        
        if n < 10:
            # Very small dataset - use different split
            train_end = max(1, n - 4)  # Leave at least 2 for val and 2 for test
            val_end = train_end + min(2, n - train_end - 2)
        else:
            train_end = int(n * TRAIN_RATIO)
            val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        
        # Ensure at least min_val_test samples in each split
        if val_end - train_end < min_val_test:
            val_end = train_end + min_val_test
        if n - val_end < min_val_test:
            val_end = n - min_val_test
        
        class_counts = {'train': 0, 'val': 0, 'test': 0}
        
        for i, img in enumerate(images):
            line = f"{cls}/{img}\n"
            
            if i < train_end:
                train_f.write(line)
                class_counts['train'] += 1
                total_counts['train'] += 1
            elif i < val_end:
                val_f.write(line)
                class_counts['val'] += 1
                total_counts['val'] += 1
            else:
                test_f.write(line)
                class_counts['test'] += 1
                total_counts['test'] += 1
        
        class_details[cls] = class_counts
        print(f"{cls:25} | Train: {class_counts['train']:3} | "
              f"Val: {class_counts['val']:3} | Test: {class_counts['test']:3}")
    
    train_f.close()
    val_f.close()
    test_f.close()
    
    print("\n" + "-"*70)
    print(f"{'TOTAL':25} | Train: {total_counts['train']:3} | "
          f"Val: {total_counts['val']:3} | Test: {total_counts['test']:3}")
    print("="*70 + "\n")
    
    # Calculate and display split percentages
    total = sum(total_counts.values())
    if total > 0:
        print("Split Percentages:")
        print(f"  Train: {total_counts['train']/total*100:.1f}%")
        print(f"  Val: {total_counts['val']/total*100:.1f}%")
        print(f"  Test: {total_counts['test']/total*100:.1f}%\n")
    
    print("✓ Data splits created successfully\n")


def verify_data_integrity():
    """
    Verify that all split files exist and are readable
    """
    print("="*70)
    print("VERIFYING DATA INTEGRITY")
    print("="*70 + "\n")
    
    all_valid = True
    
    for split_name in ['train', 'val', 'test']:
        split_file = os.path.join(SPLITS_DIR, f'{split_name}.txt')
        
        if not os.path.exists(split_file):
            print(f"✗ Missing split file: {split_file}")
            all_valid = False
            continue
        
        total = 0
        corrupted = 0
        missing_files = []
        
        with open(split_file, 'r') as f:
            for line in f:
                total += 1
                cls, img_name = line.strip().split('/')
                img_path = os.path.join(PROCESSED_DIR, cls, img_name)
                
                if not os.path.exists(img_path):
                    corrupted += 1
                    missing_files.append(img_path)
        
        if corrupted == 0:
            print(f"✓ {split_name}.txt - All {total} files intact")
        else:
            print(f"✗ {split_name}.txt - {corrupted}/{total} corrupted entries")
            all_valid = False
            if corrupted <= 5:  # Show first 5 missing files
                for missing in missing_files[:5]:
                    print(f"  Missing: {missing}")
    
    print("\n" + "="*70)
    if all_valid:
        print("✓ ALL DATA VERIFIED - READY FOR TRAINING")
    else:
        print("✗ DATA INTEGRITY ISSUES FOUND - PLEASE FIX BEFORE TRAINING")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print("PLANT HEALTH MONITORING - PREPROCESSING PIPELINE")
    print("="*70)
    
    # Run preprocessing pipeline
    preprocess_images()
    create_splits()
    verify_data_integrity()
    
    print("="*70)
    print("✓ PREPROCESSING PIPELINE COMPLETED!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review the data splits above")
    print("  2. If class imbalance exists, consider using class weights")
    print("  3. Run: python train.py")
    print("="*70 + "\n")