import os
import cv2
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import load_model

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GradCAMExplainer:
    """
    Grad-CAM visualization for model explainability using pytorch-grad-cam
    """
    
    def __init__(self, model_path=None):
        """
        Initialize Grad-CAM explainer
        
        Args:
            model_path: Path to trained model
        """
        if model_path is None:
            model_path = f"{config['paths']['models']}/best_model.pth"
        
        if not os.path.exists(model_path):
            model_path = f"{config['paths']['models']}/model_final.pth"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        print(f"■ Loading model from: {model_path}")
        self.model, _ = load_model(model_path, device)
        self.model.eval()
        
        self.img_size = config['image']['size']
        self.class_names = config['classes']
        
        # Define preprocessing
        self.transform = A.Compose([
            A.Normalize(
                mean=config['image']['normalize_mean'],
                std=config['image']['normalize_std']
            ),
            ToTensorV2()
        ])
        
        # Find target layer for Grad-CAM
        self.target_layer = self._find_target_layer()
        
        # Initialize Grad-CAM
        self.cam = GradCAM(model=self.model, target_layers=[self.target_layer])
        
        print(f"✓ Model loaded")
        print(f"  Target layer: {self.target_layer}")
    
    def _find_target_layer(self):
        """
        Automatically find the last convolutional layer in the model
        """
        # For EfficientNet and similar models, target the last conv layer
        if hasattr(self.model.base_model, 'blocks'):
            # EfficientNet structure
            return self.model.base_model.blocks[-1]
        elif hasattr(self.model.base_model, 'stages'):
            # ConvNeXt structure
            return self.model.base_model.stages[-1]
        elif hasattr(self.model.base_model, 'layer4'):
            # ResNet structure
            return self.model.base_model.layer4[-1]
        else:
            raise ValueError("Could not find suitable target layer for Grad-CAM")
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess image
        
        Args:
            image_path: Path to input image
        
        Returns:
            Original RGB image, Preprocessed tensor, Original BGR image
        """
        # Read image
        img_bgr = cv2.imread(image_path)
        
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to RGB for processing
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size),
                                interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize for display (0-1 range)
        img_normalized_display = img_resized.astype(np.float32) / 255.0
        
        # Apply preprocessing for model
        img_tensor = self.transform(image=img_resized)['image']
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_normalized_display, img_tensor, img_bgr
    
    def generate_gradcam(self, image_path, class_idx=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            image_path: Path to input image
            class_idx: Target class index (if None, use predicted class)
        
        Returns:
            Heatmap array, predicted class, confidence
        """
        # Preprocess image
        img_display, img_tensor, _ = self.preprocess_image(image_path)
        img_tensor = img_tensor.to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_idx].item()
        
        # Use predicted class if not specified
        if class_idx is None:
            class_idx = predicted_class_idx
        
        # Generate Grad-CAM
        targets = [class_idx]
        grayscale_cam = self.cam(input_tensor=img_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        return grayscale_cam, predicted_class_idx, confidence
    
    def visualize_gradcam(self, image_path, output_path=None, alpha=0.4):
        """
        Create and save Grad-CAM visualization
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization (optional)
            alpha: Transparency of heatmap overlay
        
        Returns:
            Path to saved visualization
        """
        # Generate heatmap
        heatmap, predicted_idx, confidence = self.generate_gradcam(image_path)
        
        # Load original image
        img_display, _, img_bgr = self.preprocess_image(image_path)
        
        # Create visualization using show_cam_on_image
        visualization = show_cam_on_image(img_display, heatmap, use_rgb=True)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(img_display)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Activation Heatmap', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # Superimposed
        axes[2].imshow(visualization)
        predicted_class = self.class_names[predicted_idx]
        axes[2].set_title(
            f'Grad-CAM: {predicted_class}\nConfidence: {confidence*100:.2f}%',
            fontsize=14, 
            fontweight='bold'
        )
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save
        if output_path is None:
            os.makedirs(f"{config['paths']['outputs']}/gradcam", exist_ok=True)
            filename = os.path.basename(image_path)
            output_path = f"{config['paths']['outputs']}/gradcam/gradcam_{filename}"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Grad-CAM saved: {output_path}")
        
        return output_path
    
    def compare_all_classes(self, image_path, output_path=None):
        """
        Generate Grad-CAM for all classes
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization
        """
        img_display, img_tensor, _ = self.preprocess_image(image_path)
        img_tensor = img_tensor.to(device)
        
        n_classes = len(self.class_names)
        fig, axes = plt.subplots(2, n_classes, figsize=(5*n_classes, 10))
        
        for class_idx in range(n_classes):
            # Generate heatmap for this class
            targets = [class_idx]
            grayscale_cam = self.cam(input_tensor=img_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Create visualization
            visualization = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)
            
            # Plot
            if n_classes > 1:
                axes[0, class_idx].imshow(grayscale_cam, cmap='jet')
                axes[0, class_idx].set_title(f'{self.class_names[class_idx]}\n(Heatmap)', 
                                            fontweight='bold')
                axes[0, class_idx].axis('off')
                
                axes[1, class_idx].imshow(visualization)
                axes[1, class_idx].set_title(f'{self.class_names[class_idx]}\n(Overlay)', 
                                            fontweight='bold')
                axes[1, class_idx].axis('off')
            else:
                axes[0].imshow(grayscale_cam, cmap='jet')
                axes[0].set_title(f'{self.class_names[class_idx]}\n(Heatmap)', 
                                 fontweight='bold')
                axes[0].axis('off')
                
                axes[1].imshow(visualization)
                axes[1].set_title(f'{self.class_names[class_idx]}\n(Overlay)', 
                                 fontweight='bold')
                axes[1].axis('off')
        
        plt.suptitle('Grad-CAM Analysis: All Classes', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save
        if output_path is None:
            os.makedirs(f"{config['paths']['outputs']}/gradcam", exist_ok=True)
            filename = os.path.basename(image_path)
            output_path = f"{config['paths']['outputs']}/gradcam/all_classes_{filename}"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Multi-class Grad-CAM saved: {output_path}")
        
        return output_path


def main():
    """Command line interface for Grad-CAM"""
    
    parser = argparse.ArgumentParser(
        description='Generate Grad-CAM visualizations for plant health predictions'
    )
    
    parser.add_argument('image_path', type=str,
                       help='Path to plant image')
    
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model (optional)')
    
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for visualization')
    
    parser.add_argument('--all_classes', action='store_true',
                       help='Generate Grad-CAM for all classes')
    
    parser.add_argument('--alpha', type=float, default=0.4,
                       help='Transparency of heatmap overlay (0-1)')
    
    args = parser.parse_args()
    
    # Initialize explainer
    try:
        explainer = GradCAMExplainer(model_path=args.model_path)
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Generate visualization
    try:
        if args.all_classes:
            explainer.compare_all_classes(args.image_path, args.output)
        else:
            explainer.visualize_gradcam(args.image_path, args.output, args.alpha)
        
        print("\n✓ Grad-CAM visualization completed!")
    
    except Exception as e:
        print(f"✗ Error generating Grad-CAM: {e}")


if __name__ == "__main__":
    main()