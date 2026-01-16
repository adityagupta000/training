from cProfile import label
import os
import cv2
import yaml
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


class PlantHealthDataset(Dataset):
    """Optimized dataset with mixup support"""

    def __init__(self, split_file, transform=None, augment=True, mixup_alpha=0.0):
        self.img_size = config['image']['size']
        self.processed_dir = config['paths']['processed_data']
        self.augment = augment
        self.mixup_alpha = mixup_alpha

        self.label_map = {cls: idx for idx, cls in enumerate(config['classes'])}
        self.classes = config['classes']

        self.data = []
        with open(split_file, 'r') as f:
            for line in f:
                cls, img_name = line.strip().split('/')
                img_path = os.path.join(self.processed_dir, cls, img_name)
                self.data.append((img_path, self.label_map[cls]))

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_transforms()

    def _get_transforms(self):
        """Consistent normalization"""
        normalize = A.Normalize(
            mean=config['image']['normalize_mean'],
            std=config['image']['normalize_std'],
            max_pixel_value=config['image']['max_pixel_value']
        )

        if not self.augment:
            return A.Compose([normalize, ToTensorV2()])

        aug_config = config['augmentation']
        aug_prob = aug_config.get('augmentation_prob', 0.65)

        return A.Compose([
            A.HorizontalFlip(p=0.5 if aug_config['horizontal_flip'] else 0),
            A.VerticalFlip(p=0.5 if aug_config['vertical_flip'] else 0),
            A.Rotate(limit=aug_config['rotation_range'], border_mode=cv2.BORDER_REFLECT, p=aug_prob * 0.7),
            A.ShiftScaleRotate(
                shift_limit=aug_config['width_shift'],
                scale_limit=aug_config['zoom_range'],
                rotate_limit=0,
                border_mode=cv2.BORDER_REFLECT,
                p=aug_prob * 0.7
            ),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, border_mode=cv2.BORDER_REFLECT, p=1),
                A.GridDistortion(num_steps=5, distort_limit=0.2, border_mode=cv2.BORDER_REFLECT, p=1),
            ], p=aug_prob * 0.25),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=aug_prob * 0.5),
            A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=20, val_shift_limit=12, p=aug_prob * 0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1),
                A.MedianBlur(blur_limit=5, p=1),
            ], p=aug_prob * 0.2),
            A.GaussNoise(var_limit=(10, 30), p=aug_prob * 0.15),
            A.CoarseDropout(
                max_holes=4, max_height=20, max_width=20,
                min_holes=1, min_height=8, min_width=8,
                fill_value=0, p=aug_prob * 0.2
            ),
            normalize,
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
    
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        augmented = self.transform(image=img)
        img_tensor = augmented['image']
    
        return img_tensor, label

    def get_labels(self):
        return [label for _, label in self.data]


def get_balanced_class_weights(split_file, method='balanced', power=1.0, max_cap=5.0):
    """
    FIXED: Balanced class weights with capping
    
    Args:
        split_file: Training split file
        method: 'balanced', 'sqrt', or 'inverse_freq'
        power: Power scaling factor
        max_cap: Maximum weight (prevents over-correction)
    
    Returns:
        torch.Tensor: Balanced class weights
    """
    class_counts = Counter()

    with open(split_file, 'r') as f:
        for line in f:
            cls = line.strip().split('/')[0]
            class_counts[cls] += 1

    # Get counts in order
    counts = np.array([class_counts.get(cls, 1) for cls in config['classes']])
    total = counts.sum()
    n_classes = len(config['classes'])

    # Calculate weights based on method
    if method == 'balanced':
        # sklearn-style balanced weights
        weights = total / (n_classes * counts)
    elif method == 'sqrt':
        # Square root scaling (gentler)
        weights = np.sqrt(total / (n_classes * counts))
    elif method == 'inverse_freq':
        # Inverse frequency
        weights = 1.0 / counts
        weights = weights / weights.sum() * n_classes
    else:
        weights = np.ones(n_classes)

    # Apply power scaling
    if power != 1.0:
        weights = weights ** power

    # CRITICAL: Cap maximum weight
    if max_cap is not None:
        weights = np.minimum(weights, max_cap)

    # Normalize to mean=1.0
    weights = weights / weights.mean()

    weights = torch.FloatTensor(weights)

    print("\n" + "="*70)
    print("BALANCED CLASS WEIGHTS (with capping)")
    print("="*70)
    print(f"Method: {method} | Power: {power} | Max Cap: {max_cap}")
    print("-"*70)

    for cls, count, weight in zip(config['classes'], counts, weights):
        ratio = count / counts.min()
        status = "■" if count < 700 else "■" if count < 1500 else "■"
        print(f"{status} {cls:25} : {count:4} samples (ratio: {ratio:4.2f}x) | Weight: {weight:.3f}")

    print("-"*70)
    print(f"Weight range: {weights.min():.3f} - {weights.max():.3f}")
    print(f"Imbalance ratio: {counts.max() / counts.min():.2f}:1")
    print("="*70 + "\n")

    return weights


def get_moderate_sampler(dataset, strategy='moderate'):
    """
    FIXED: Moderate weighted sampler (prevents over-sampling)
    
    Args:
        dataset: PyTorch dataset
        strategy: 'conservative', 'moderate', or 'aggressive'
    
    Returns:
        WeightedRandomSampler
    """
    labels = dataset.get_labels()
    class_counts = Counter(labels)

    # Calculate base weights
    base_weights = {label: 1.0 / count for label, count in class_counts.items()}

    # Apply strategy
    if strategy == 'conservative':
        # Gentle oversampling
        power = 0.5  # Square root
    elif strategy == 'moderate':
        # Moderate oversampling
        power = 0.75
    elif strategy == 'aggressive':
        # Strong oversampling
        power = 1.0
    else:
        power = 0.75

    # Apply power scaling and cap
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())

    adjusted_weights = {}
    for label, count in class_counts.items():
        # Scale weight based on strategy
        weight = (max_count / count) ** power
        # Cap to prevent extreme oversampling
        weight = min(weight, 5.0)
        adjusted_weights[label] = weight

    # Create sample weights
    sample_weights = [adjusted_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    print(f"✓ {strategy.upper()} weighted sampler created")
    print(f"  Sampling power: {power}")
    print(f"  Weight range: {min(adjusted_weights.values()):.2f} - {max(adjusted_weights.values()):.2f}")

    # Show effective sampling rates
    print("\n  Effective sampling rates:")
    for cls_idx, cls_name in enumerate(config['classes']):
        if cls_idx in adjusted_weights:
            count = class_counts[cls_idx]
            weight = adjusted_weights[cls_idx]
            effective_count = count * weight
            print(f"  {cls_name:25} : {count:4} → ~{effective_count:6.0f} per epoch")
    print()

    return sampler


def mixup_data(x, y, alpha=0.2):
    """
    Mixup augmentation
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def get_data_loaders(batch_size=None, num_workers=None, use_weighted_sampler=True):
    """Create optimized data loaders"""

    if batch_size is None:
        batch_size = config['training']['batch_size']

    if num_workers is None:
        num_workers = config['training'].get('num_workers', 4)

    splits_dir = config['paths']['splits']

    # Mixup alpha
    mixup_alpha = config['training'].get('mixup_alpha', 0.0)

    # Datasets
    train_dataset = PlantHealthDataset(
        split_file=os.path.join(splits_dir, 'train.txt'),
        augment=True,
        mixup_alpha=mixup_alpha
    )

    val_dataset = PlantHealthDataset(
        split_file=os.path.join(splits_dir, 'val.txt'),
        augment=False
    )

    test_dataset = PlantHealthDataset(
        split_file=os.path.join(splits_dir, 'test.txt'),
        augment=False
    )

    # Weighted sampler
    train_sampler = None
    shuffle = True

    if use_weighted_sampler and config['training'].get('oversample_minority', True):
        strategy = config['training'].get('sampling_strategy', 'moderate')
        train_sampler = get_moderate_sampler(train_dataset, strategy=strategy)
        shuffle = False

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing balanced data loader...")

    train_loader, val_loader, test_loader = get_data_loaders(batch_size=4)

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Pixel range: [{images.min():.2f}, {images.max():.2f}]")

    print("\n✓ Balanced data loader working!")
    print("✓ Moderate oversampling prevents majority dominance")
    print("✓ Weight capping prevents minority over-prediction")