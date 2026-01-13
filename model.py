import yaml
import torch
import torch.nn as nn
import timm

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


class PlantHealthModel(nn.Module):
    """
    Enhanced Plant Health Model with:
    - Reduced dropout for minority class stability
    - Proper BatchNorm handling
    - Label smoothing support
    """

    def __init__(self, model_name=None, num_classes=None, pretrained=True, dropout=None):
        super(PlantHealthModel, self).__init__()

        if model_name is None:
            model_name = config['model']['name']
        if num_classes is None:
            num_classes = config['model']['num_classes']
        if dropout is None:
            dropout = config['model']['dropout']

        self.model_name = model_name
        self.num_classes = num_classes

        # Normalize model name (remove tf_ prefix if present)
        timm_model_name = model_name.replace('tf_', '')

        # Load base model using timm
        try:
            self.base_model = timm.create_model(
                timm_model_name, 
                pretrained=pretrained, 
                num_classes=0
            )
            feature_dim = self.base_model.num_features
        except Exception as e:
            raise ValueError(
                f"Could not load model '{timm_model_name}' (original: '{model_name}'). "
                f"Error: {e}\n"
                f"Try one of these common models:\n"
                f"  - efficientnetv2_b0, efficientnetv2_b1, efficientnetv2_b2, efficientnetv2_b3\n"
                f"  - convnext_tiny, convnext_small\n"
                f"  - resnet50, resnet101\n"
                f"  - tf_efficientnetv2_b0, tf_efficientnetv2_b2 (will auto-convert to efficientnetv2)"
            )

        # FIXED: Simpler, more stable classification head
        # Reduced dropout prevents feature loss for minority classes
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),

            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.5),  # Progressive reduction
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.3),  # Even lower for final layer
            nn.Linear(256, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

        print(f"\n✓ Model architecture: {timm_model_name}")
        print(f"✓ Pretrained weights: {'ImageNet' if pretrained else 'Random'}")
        print(f"✓ Number of classes: {num_classes}")
        print(f"✓ Feature dimension: {feature_dim}")
        print(f"✓ Dropout rate: {dropout} (REDUCED for minority class stability)")

    def _initialize_weights(self):
        """Initialize classifier weights with Xavier initialization"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass"""
        features = self.base_model(x)
        output = self.classifier(features)
        return output

    def freeze_base(self):
        """Freeze base model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = False
        print("\n✓ Base model frozen")

    def unfreeze_base(self, num_layers=None):
        """
        Unfreeze base model for fine-tuning
        FIXED: More gradual unfreezing
        """
        if num_layers is None:
            num_layers = config['model']['unfreeze_layers']

        # Unfreeze all first
        for param in self.base_model.parameters():
            param.requires_grad = True

        all_layers = list(self.base_model.children())
        total_layers = len(all_layers)

        # Freeze early layers
        freeze_until = max(0, total_layers - num_layers)
        for i, layer in enumerate(all_layers[:freeze_until]):
            for param in layer.parameters():
                param.requires_grad = False

        trainable_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.base_model.parameters())

        print(f"\n■ Base Model Unfreezing:")
        print(f"  Total layers: {total_layers}")
        print(f"  Unfroze last {num_layers} layers")
        print(f"  Trainable: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Focuses learning on hard, misclassified examples
    Down-weights easy examples (high confidence correct predictions)
    """

    def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) raw logits
            targets: (N,) class indices
        """
        ce_loss = nn.functional.cross_entropy(
            inputs, targets,
            weight=self.weight,
            reduction='none'
        )

        # Get probabilities
        p = torch.exp(-ce_loss)

        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - p) ** self.gamma

        # Apply alpha weighting
        if self.alpha is not None:
            focal_weight = self.alpha * focal_weight

        # Final loss
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    Prevents overconfident predictions
    Improves calibration and generalization
    """

    def __init__(self, epsilon=0.1, weight=None, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) raw logits
            targets: (N,) class indices
        """
        n_classes = inputs.size(1)
        log_probs = nn.functional.log_softmax(inputs, dim=1)

        # Create smoothed targets
        # True class gets (1 - epsilon) probability
        # Other classes share epsilon probability
        targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * (1 - self.epsilon) + self.epsilon / n_classes

        # Calculate loss
        loss = -(targets_smooth * log_probs).sum(dim=1)

        # Apply class weights if provided
        if self.weight is not None:
            loss = loss * self.weight[targets]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def build_model(model_name=None, num_classes=None, pretrained=True):
    """Build the classification model"""
    model = PlantHealthModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained
    )

    # Freeze base initially
    model.freeze_base()

    return model


def print_model_summary(model, input_size=(3, 224, 224)):
    """Print model summary"""
    print("\n" + "="*70)
    print("MODEL SUMMARY")
    print("="*70 + "\n")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: {model.model_name}")
    print(f"Input shape: {input_size}")
    print(f"Output classes: {model.num_classes}")
    print(f"\n■ Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")
    print(f"  Trainable %: {trainable_params/total_params*100:.2f}%")
    print("="*70 + "\n")


def save_model(model, optimizer, epoch, best_metric, save_path, metric_name='accuracy'):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        'metric_name': metric_name,
        'model_name': model.model_name,
        'num_classes': model.num_classes,
        'config': config
    }
    torch.save(checkpoint, save_path)
    print(f"✓ Model saved: {save_path}")


def load_model(load_path, device='cuda'):
    """
    Safely load old PyTorch checkpoints with numpy types in PyTorch >=2.6
    """
    import numpy as np

    # Allowlist numpy globals for safe unpickling
    safe_globals = [
        np.dtype,
        np.int64,
        np.float32,
        np.float64,
        np.bool_,
        np.core.multiarray.scalar
    ]

    # Use safe_globals context
    with torch.serialization.safe_globals(safe_globals):
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)

    # Build model
    model = PlantHealthModel(
        model_name=checkpoint['model_name'],
        num_classes=checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    metric_name = checkpoint.get('metric_name', 'accuracy')
    print(f"✓ Model loaded: {load_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best {metric_name}: {checkpoint['best_metric']:.4f}")

    return model, checkpoint

    
if __name__ == "__main__":
    print("Testing model architecture...")

    model = build_model()
    print_model_summary(model)

    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"✓ Forward pass successful!")
    print(f"  Output shape: {output.shape}")

    # Test focal loss
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    dummy_target = torch.tensor([0])
    loss = focal(output, dummy_target)
    print(f"✓ Focal loss: {loss.item():.4f}")

    print("\n✓ All tests passed!")