import os
import cv2
import yaml
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def analyze_raw_dataset():
    """
    Comprehensive analysis of the raw dataset
    """
    print("\n" + "="*70)
    print("DATASET ANALYSIS TOOL")
    print("="*70 + "\n")
    
    raw_dir = config['paths']['raw_data']
    
    if not os.path.exists(raw_dir):
        print(f"✗ Raw data directory not found: {raw_dir}")
        return
    
    class_stats = {}
    total_images = 0
    file_extensions = Counter()
    image_sizes = []
    corrupted = []
    
    # Analyze each class
    for cls in os.listdir(raw_dir):
        cls_path = os.path.join(raw_dir, cls)
        
        if not os.path.isdir(cls_path):
            continue
        
        print(f"■ Analyzing: {cls}")
        
        # Collect all images recursively
        images = []
        subdirs = []
        
        for root, dirs, files in os.walk(cls_path):
            if root != cls_path:
                subdir_name = os.path.relpath(root, cls_path)
                subdirs.append(subdir_name)
            
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    images.append(os.path.join(root, file))
                    file_extensions[ext] += 1
        
        # Analyze images
        valid_count = 0
        sizes = []
        
        for img_path in images:
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    sizes.append((img.shape[0], img.shape[1]))
                    valid_count += 1
                else:
                    corrupted.append(img_path)
            except:
                corrupted.append(img_path)
        
        class_stats[cls] = {
            'total': len(images),
            'valid': valid_count,
            'corrupted': len(images) - valid_count,
            'subdirs': subdirs,
            'sizes': sizes
        }
        
        total_images += valid_count
        image_sizes.extend(sizes)
        
        # Print class summary
        print(f"  Total images: {len(images)}")
        print(f"  Valid: {valid_count}")
        if len(images) - valid_count > 0:
            print(f"  Corrupted: {len(images) - valid_count}")
        if subdirs:
            print(f"  Subdirectories: {', '.join(subdirs)}")
        if sizes:
            heights, widths = zip(*sizes)
            print(f"  Size range: {min(widths)}×{min(heights)} to {max(widths)}×{max(heights)}")
        print()
    
    # Overall statistics
    print("="*70)
    print("OVERALL STATISTICS")
    print("="*70 + "\n")
    
    # Class distribution
    print("Class Distribution:")
    for cls, stats in sorted(class_stats.items(), key=lambda x: x[1]['valid'], reverse=True):
        percentage = (stats['valid'] / total_images * 100) if total_images > 0 else 0
        bar_length = int(percentage / 2)
        bar = '■' * bar_length
        print(f"  {cls:25} : {stats['valid']:4} ({percentage:5.1f}%) {bar}")
    print(f"  {'TOTAL':25} : {total_images:4}\n")
    
    # Class imbalance
    if class_stats:
        counts = [s['valid'] for s in class_stats.values()]
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else 0
        
        print(f"Class Imbalance Ratio: {imbalance_ratio:.2f}x")
        if imbalance_ratio > 3:
            print("  ■ HIGH IMBALANCE - Class weights will be used during training")
        elif imbalance_ratio > 1.5:
            print("  ■ MODERATE IMBALANCE - Class weights recommended")
        else:
            print("  ✓ Well balanced\n")
        print()
    
    # File extensions
    print("File Extensions:")
    for ext, count in file_extensions.most_common():
        print(f"  {ext:10} : {count:4} files")
    print()
    
    # Image sizes
    if image_sizes:
        heights, widths = zip(*image_sizes)
        print("Image Size Statistics:")
        print(f"  Width: min={min(widths):4}, max={max(widths):4}, avg={int(np.mean(widths)):4}")
        print(f"  Height: min={min(heights):4}, max={max(heights):4}, avg={int(np.mean(heights)):4}")
        
        # Check for very small images
        small_images = sum(1 for h, w in image_sizes if h < 100 or w < 100)
        if small_images > 0:
            print(f"  ■ {small_images} images smaller than 100×100 pixels")
        print()
    
    # Corrupted files
    if corrupted:
        print(f"■ Corrupted Files: {len(corrupted)}")
        for corrupt_file in corrupted[:5]:
            print(f"  - {corrupt_file}")
        if len(corrupted) > 5:
            print(f"  ... and {len(corrupted) - 5} more")
        print()
    
    # Recommendations
    print("="*70)
    print("RECOMMENDATIONS")
    print("="*70 + "\n")
    
    recommendations = []
    
    # Check dataset size
    if total_images < 500:
        recommendations.append("■ Dataset is very small (<500 images)")
        recommendations.append("  → Collect at least 500 more images")
    elif total_images < 2000:
        recommendations.append("■ Dataset is small (<2000 images)")
        recommendations.append("  → Heavy augmentation will be used")
        recommendations.append("  → Expect 75-85% accuracy")
    else:
        recommendations.append("✓ Dataset size is adequate")
    
    # Check class balance
    if class_stats:
        counts = [s['valid'] for s in class_stats.values()]
        min_count = min(counts)
        
        if min_count < 50:
            recommendations.append("■ Some classes have <50 samples")
            recommendations.append("  → Critical: Collect more data for small classes")
        elif min_count < 100:
            recommendations.append("■ Some classes have <100 samples")
            recommendations.append("  → Recommended: Collect more data")
        
        if imbalance_ratio > 3:
            recommendations.append("■ High class imbalance detected")
            recommendations.append("  → Class weights will be automatically applied")
    
    # Check for subdirectories
    has_subdirs = any(stats['subdirs'] for stats in class_stats.values())
    if has_subdirs:
        recommendations.append("✓ Subdirectories detected (will be flattened automatically)")
    
    if not recommendations:
        recommendations.append("✓ Dataset looks good!")
    
    for rec in recommendations:
        print(rec)
    
    print("\n" + "="*70)
    print("Next Steps:")
    print("  1. Run: python preprocessing.py")
    print("  2. Then: python train.py")
    print("="*70 + "\n")


def visualize_class_distribution():
    """
    Create a bar plot of class distribution
    """
    raw_dir = config['paths']['raw_data']
    
    class_counts = {}
    
    for cls in os.listdir(raw_dir):
        cls_path = os.path.join(raw_dir, cls)
        
        if not os.path.isdir(cls_path):
            continue
        
        count = 0
        for root, dirs, files in os.walk(cls_path):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    count += 1
        
        class_counts[cls] = count
    
    if not class_counts:
        print("No data to visualize")
        return
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = ['#2ecc71' if c > 150 else '#f39c12' if c > 80 else '#e74c3c' for c in counts]
    
    bars = ax.bar(classes, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('Dataset Class Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.7, label='Good (>150)'),
        Patch(facecolor='#f39c12', alpha=0.7, label='Fair (80-150)'),
        Patch(facecolor='#e74c3c', alpha=0.7, label='Low (<80)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/dataset_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Class distribution plot saved: outputs/dataset_distribution.png")
    plt.close()


if __name__ == "__main__":
    analyze_raw_dataset()
    
    try:
        visualize_class_distribution()
    except Exception as e:
        print(f"\nNote: Could not create visualization: {e}")
        print("This is normal if matplotlib has display issues.")