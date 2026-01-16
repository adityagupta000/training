import os
import cv2
import yaml
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm import tqdm

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Configuration
MIN_SAMPLES_THRESHOLD = 500  # Target minimum samples per class
AUGMENTATIONS_PER_IMAGE = 5  # How many augmented versions to create per original image


def get_augmentation_pipeline():
    """
    Heavy augmentation pipeline for generating diverse synthetic samples
    """
    return A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=40, border_mode=cv2.BORDER_REFLECT, p=0.8),
        A.ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=30,
            border_mode=cv2.BORDER_REFLECT,
            p=0.8
        ),
        
        # Advanced geometric
        A.OneOf([
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                border_mode=cv2.BORDER_REFLECT,
                p=1
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                border_mode=cv2.BORDER_REFLECT,
                p=1
            ),
            A.OpticalDistortion(
                distort_limit=0.1,
                shift_limit=0.1,
                border_mode=cv2.BORDER_REFLECT,
                p=1
            ),
        ], p=0.5),
        
        # Color augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),
        A.HueSaturationValue(
            hue_shift_limit=25,
            sat_shift_limit=35,
            val_shift_limit=25,
            p=0.7
        ),
        A.RGBShift(
            r_shift_limit=25,
            g_shift_limit=25,
            b_shift_limit=25,
            p=0.5
        ),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.3,
            hue=0.15,
            p=0.5
        ),
        
        # Blur and noise
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 9), p=1),
            A.MedianBlur(blur_limit=7, p=1),
            A.MotionBlur(blur_limit=9, p=1),
        ], p=0.4),
        
        A.OneOf([
            A.GaussNoise(var_limit=(10, 70), p=1),
            A.ISONoise(color_shift=(0.01, 0.08), intensity=(0.1, 0.6), p=1),
            A.MultiplicativeNoise(multiplier=(0.85, 1.15), p=1),
        ], p=0.4),
        
        # Weather effects
        A.OneOf([
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=3,
                shadow_dimension=5,
                p=1
            ),
            A.RandomFog(
                fog_coef_lower=0.1,
                fog_coef_upper=0.4,
                alpha_coef=0.1,
                p=1
            ),
            A.RandomRain(
                slant_lower=-15,
                slant_upper=15,
                drop_length=25,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=3,
                brightness_coefficient=0.9,
                rain_type="default",
                p=1
            ),
        ], p=0.3),
        
        # Quality degradation
        A.OneOf([
            A.Downscale(scale_min=0.5, scale_max=0.8, p=1),
            A.ImageCompression(quality_lower=50, quality_upper=90, p=1),
        ], p=0.3),
        
        # Cutout
        A.CoarseDropout(
            max_holes=10,
            max_height=40,
            max_width=40,
            min_holes=1,
            min_height=10,
            min_width=10,
            fill_value=0,
            p=0.4
        ),
    ])


def augment_minority_classes():
    """
    Augment classes that have fewer samples than the threshold
    """
    raw_dir = config['paths']['raw_data']
    
    print("\n" + "="*70)
    print("MINORITY CLASS AUGMENTATION")
    print("="*70 + "\n")
    print(f"Target: {MIN_SAMPLES_THRESHOLD} samples per class")
    print(f"Augmentations per image: {AUGMENTATIONS_PER_IMAGE}\n")
    
    # Get class counts
    class_counts = {}
    for cls in os.listdir(raw_dir):
        cls_path = os.path.join(raw_dir, cls)
        
        if not os.path.isdir(cls_path):
            continue
        
        # Count images recursively
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
        count = 0
        for root, dirs, files in os.walk(cls_path):
            for file in files:
                if Path(file).suffix in image_extensions:
                    count += 1
        
        class_counts[cls] = count
    
    # Find classes that need augmentation
    classes_to_augment = {
        cls: count for cls, count in class_counts.items() 
        if count < MIN_SAMPLES_THRESHOLD
    }
    
    if not classes_to_augment:
        print("✓ All classes have sufficient samples!")
        print("\nCurrent distribution:")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls:25} : {count:4} samples")
        return
    
    print("Classes requiring augmentation:")
    for cls, count in classes_to_augment.items():
        needed = MIN_SAMPLES_THRESHOLD - count
        print(f"  {cls:25} : {count:4} samples → Need {needed:4} more")
    print()
    
    # Create augmentation pipeline
    transform = get_augmentation_pipeline()
    
    # Augment each minority class
    for cls, original_count in classes_to_augment.items():
        print(f"\n■ Augmenting: {cls}")
        
        cls_path = os.path.join(raw_dir, cls)
        
        # Collect all existing images
        image_files = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
        
        for root, dirs, files in os.walk(cls_path):
            for file in files:
                if Path(file).suffix in image_extensions:
                    full_path = os.path.join(root, file)
                    image_files.append(full_path)
        
        # Create augmented subdirectory
        aug_dir = os.path.join(cls_path, 'augmented')
        os.makedirs(aug_dir, exist_ok=True)
        
        # Calculate how many augmentations we need
        samples_needed = MIN_SAMPLES_THRESHOLD - original_count
        
        # Distribute augmentations across all images
        augmentations_needed = samples_needed
        images_to_augment = image_files * (augmentations_needed // len(image_files) + 1)
        images_to_augment = images_to_augment[:augmentations_needed]
        
        print(f"  Original samples: {original_count}")
        print(f"  Creating: {len(images_to_augment)} augmented samples")
        print(f"  Target total: {MIN_SAMPLES_THRESHOLD}")
        
        # Generate augmented images
        successful = 0
        failed = 0
        
        for idx, img_path in enumerate(tqdm(images_to_augment, desc="  Generating")):
            try:
                # Read image
                img = cv2.imread(img_path)
                
                if img is None:
                    failed += 1
                    continue
                
                # Convert BGR to RGB for augmentation
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Apply augmentation
                augmented = transform(image=img_rgb)
                aug_img = augmented['image']
                
                # Convert back to BGR for saving
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                
                # Generate unique filename
                original_name = Path(img_path).stem
                aug_filename = f"{original_name}_aug_{idx:04d}.jpg"
                aug_path = os.path.join(aug_dir, aug_filename)
                
                # Save with high quality
                cv2.imwrite(aug_path, aug_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                successful += 1
                
            except Exception as e:
                failed += 1
                continue
        
        print(f"  ✓ Created: {successful} augmented images")
        if failed > 0:
            print(f"  ✗ Failed: {failed} images")
        
        # Verify final count
        new_count = 0
        for root, dirs, files in os.walk(cls_path):
            for file in files:
                if Path(file).suffix in image_extensions:
                    new_count += 1
        
        print(f"  ✓ Final total: {new_count} samples")
    
    print("\n" + "="*70)
    print("AUGMENTATION SUMMARY")
    print("="*70)
    
    # Show final distribution
    for cls in os.listdir(raw_dir):
        cls_path = os.path.join(raw_dir, cls)
        
        if not os.path.isdir(cls_path):
            continue
        
        # Count images
        count = 0
        for root, dirs, files in os.walk(cls_path):
            for file in files:
                if Path(file).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                    count += 1
        
        original = class_counts.get(cls, 0)
        added = count - original
        
        if added > 0:
            print(f"{cls:25} : {original:4} → {count:4} (+{added:4} augmented)")
        else:
            print(f"{cls:25} : {count:4} (unchanged)")
    
    print("="*70 + "\n")
    print("✓ Augmentation completed!")
    print("\nNext steps:")
    print("  1. Run: python analyze_dataset.py (to verify new distribution)")
    print("  2. Run: python preprocessing.py")
    print("  3. Run: python train.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    augment_minority_classes()