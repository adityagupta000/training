"""
Updated Inference Script with Temperature Scaling Support

CHANGES FROM ORIGINAL:
1. Automatically detects if model has temperature calibration
2. Uses calibrated predictions when available
3. Adds calibration status to output
4. Backward compatible with non-calibrated models

Usage remains the same:
    python inference.py <image_path>
    python inference.py <image_path> --explain
"""

import os
from unittest import result
import cv2
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import load_model

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PlantHealthPredictor:
    """
    UPDATED: Predictor with automatic temperature scaling support
    """
    
    def __init__(self, model_path=None):
        if model_path is None:
            # Try calibrated model first, fall back to regular
            calibrated_path = f"{config['paths']['models']}/calibrated_model.pth"
            regular_path = f"{config['paths']['models']}/best_model.pth"
            
            if os.path.exists(calibrated_path):
                model_path = calibrated_path
                print("■ Using calibrated model (improved confidence scores)")
            elif os.path.exists(regular_path):
                model_path = regular_path
                print("■ Using regular model (run calibrate_confidence.py for better confidence)")
            else:
                model_path = f"{config['paths']['models']}/model_final.pth"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"■ Loading model: {model_path}")
        self.model, self.checkpoint = load_model(model_path, device)
        self.model.eval()
        
        # Check for temperature calibration
        self.temperature = self.checkpoint.get('temperature', 1.0)
        self.is_calibrated = self.checkpoint.get('calibrated', False)
        
        if self.is_calibrated:
            print(f"✓ Model is calibrated (Temperature: {self.temperature:.4f})")
            print("  → Confidence scores are more reliable")
        else:
            print("■ Model is not calibrated")
            print("  → Run: python calibrate_confidence.py")
        
        self.img_size = config['image']['size']
        self.class_names = config['classes']
        
        # EXACT same normalization as training
        self.transform = A.Compose([
            A.Normalize(
                mean=config['image']['normalize_mean'],
                std=config['image']['normalize_std'],
                max_pixel_value=config['image']['max_pixel_value']
            ),
            ToTensorV2()
        ])
        
        print(f"✓ Model loaded")
        print(f" Classes: {', '.join(self.class_names)}")
    
    def preprocess_image(self, image_path):
        """
        Consistent preprocessing
        """
        # Read image
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize with same interpolation as training
        img = cv2.resize(img, (self.img_size, self.img_size),
                        interpolation=cv2.INTER_LANCZOS4)
        
        # Apply EXACT same normalization as training
        img = self.transform(image=img)['image']
        
        # Add batch dimension
        img = img.unsqueeze(0)
        
        return img
    
    def predict(self, image_path, return_probabilities=False, 
                confidence_threshold=None):
        """
        UPDATED: Prediction with automatic temperature scaling
        """
        if confidence_threshold is None:
            confidence_threshold = config['evaluation'].get('default_threshold', 0.5)
        
        # Preprocess
        img = self.preprocess_image(image_path)
        img = img.to(device)
        
        # Predict with temperature scaling
        with torch.no_grad():
            logits = self.model(img)
            
            # Apply temperature scaling if calibrated
            if self.is_calibrated:
                scaled_logits = logits / self.temperature
            else:
                scaled_logits = logits
            
            probabilities = torch.softmax(scaled_logits, dim=1)[0]
        
        probabilities = probabilities.cpu().numpy()
        
        # Get prediction
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # Check threshold
        low_confidence = confidence < confidence_threshold
        
        # Category mapping
        if predicted_class.startswith("Pest_"):
            category = "Pest"
            subtype = predicted_class.replace("Pest_", "")
        elif predicted_class.startswith("Nutrient_"):
            category = "Nutrient Deficiency"
            subtype = predicted_class.replace("Nutrient_", "")
        elif predicted_class == "Not_Plant":  
            category = "Invalid Input"  
            subtype = None
        else:
            category = predicted_class
            subtype = None
        
        result = {
            'predicted_class': predicted_class,
            'category': category,
            'subtype': subtype,
            'confidence': confidence,
            'confidence_percentage': confidence * 100,
            'low_confidence': low_confidence,
            'threshold': confidence_threshold,
            'is_calibrated': self.is_calibrated,  # NEW
            'temperature': self.temperature  # NEW
        }
        
        if return_probabilities:
            result['all_probabilities'] = {
                name: float(prob)
                for name, prob in zip(self.class_names, probabilities)
            }
        
        # Top 3 predictions
        top3_idx = np.argsort(probabilities)[-3:][::-1]
        result['top3'] = [
            {
                'class': self.class_names[idx],
                'confidence': float(probabilities[idx])
            }
            for idx in top3_idx
        ]
        
        return result
    
    def predict_with_explanation(self, image_path):
        """Enhanced prediction with explanations"""
        result = self.predict(image_path, return_probabilities=True)
        
        # Complete explanations
        explanations = {
            "Healthy": "Plant appears healthy. No visible issues detected.",
            "Pest_Fungal": "Fungal infection detected. Look for powdery spots, mold, or discoloration. Treatment: fungicide.",
            "Pest_Bacterial": "Bacterial infection detected. Water-soaked lesions or wilting. Treatment: copper-based spray.",
            "Pest_Insect": "Insect damage detected. Holes, chewed edges, or insect presence. Treatment: insecticide or natural predators.",
            "Nutrient_Nitrogen": "Nitrogen deficiency. Yellowing of older leaves, stunted growth. Treatment: nitrogen fertilizer.",
            "Nutrient_Potassium": "Potassium deficiency. Leaf edge browning, weak stems. Treatment: potassium fertilizer.",
            "Water_Stress": "Water stress detected. Wilting or dry soil. Treatment: adjust watering schedule.",
            "Not_Plant": "This is not a plant image. Please upload a clear image of a plant leaf for disease detection."
        }
        
        result['explanation'] = explanations.get(
            result['predicted_class'],
            "Unable to provide specific explanation."
        )
        
        # UPDATED: Confidence interpretation (calibrated models more reliable)
        conf = result['confidence']
        
        if self.is_calibrated:
            if conf > 0.80:
                result['confidence_level'] = "Very High"
                result['reliability'] = "Highly reliable - calibrated confidence"
            elif conf > 0.65:
                result['confidence_level'] = "High"
                result['reliability'] = "Reliable - calibrated confidence"
            elif conf > 0.50:
                result['confidence_level'] = "Moderate"
                result['reliability'] = "Moderately reliable - consider context"
            else:
                result['confidence_level'] = "Low"
                result['reliability'] = "Low confidence - manual inspection recommended"
   
                if result['predicted_class'] == 'Not_Plant':
                    result['warning'] = "■ Non-plant image detected. Please upload a plant leaf image."
                else:
                    result['warning'] = "■ Low confidence prediction. Consider retaking image or consulting expert."
        else:
            # Uncalibrated (original thresholds - less reliable)
            if conf > 0.85:
                result['confidence_level'] = "Very High"
                result['reliability'] = "Likely reliable (uncalibrated - may be overconfident)"
            elif conf > 0.70:
                result['confidence_level'] = "High"
                result['reliability'] = "Moderately reliable (uncalibrated)"
            elif conf > 0.55:
                result['confidence_level'] = "Moderate"
                result['reliability'] = "Uncertain (uncalibrated - calibration recommended)"
            else:
                result['confidence_level'] = "Low"
                result['reliability'] = "Low confidence - manual inspection recommended"
                if result['predicted_class'] == 'Not_Plant':
                    result['warning'] = "■ Non-plant image detected. Please upload a plant leaf image."
                else:
                    result['warning'] = "■ Low confidence. Model not calibrated - run calibrate_confidence.py"
        
        return result


def print_prediction_result(result):
    """Pretty print prediction with calibration info"""
    print("\n" + "="*70)
    print("PREDICTION RESULT")
    print("="*70)
    
    if 'error' in result:
        print(f"\n✗ Error: {result['error']}")
        return
    
    # Calibration status
    if result.get('is_calibrated', False):
        print(f"\n✓ Using calibrated model (T={result['temperature']:.4f})")
        print("  Confidence scores are more reliable")
    else:
        print(f"\n■ Using uncalibrated model")
        print("  Tip: Run calibrate_confidence.py for better confidence scores")
    
    print(f"\nPredicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence_percentage']:.2f}%")
    print(f"Level: {result.get('confidence_level', 'N/A')}")
    
    if result.get('low_confidence'):
        print(f"\n⚠ Confidence below threshold ({result['threshold']*100:.0f}%)")
    
    if 'explanation' in result:
        print(f"\nExplanation:")
        print(f" {result['explanation']}")
    
    if 'reliability' in result:
        print(f"\nReliability: {result['reliability']}")
    
    if 'warning' in result:
        print(f"\n{result['warning']}")
    
    # Top 3
    if 'top3' in result:
        print(f"\nTop 3 Predictions:")
        for i, pred in enumerate(result['top3'], 1):
            print(f" {i}. {pred['class']:25} {pred['confidence']*100:5.2f}%")
    
    # All probabilities
    if 'all_probabilities' in result:
        print(f"\nAll Class Probabilities:")
        for cls, prob in result['all_probabilities'].items():
            bar_len = int(prob * 40)
            bar = '█' * bar_len + '░' * (40 - bar_len)
            print(f" {cls:25} {bar} {prob*100:5.2f}%")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plant Health Inference')
    
    parser.add_argument('image_path', type=str, help='Path to plant image')
    parser.add_argument('--model_path', type=str, default=None, help='Model path')
    parser.add_argument('--explain', action='store_true', help='Detailed explanation')
    parser.add_argument('--save', action='store_true', help='Save result to JSON')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Confidence threshold (default: 0.55)')
    
    args = parser.parse_args()
    
    # Initialize predictor (will auto-detect calibrated model)
    try:
        predictor = PlantHealthPredictor(model_path=args.model_path)
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Predict
    try:
        if args.explain:
            result = predictor.predict_with_explanation(args.image_path)
        else:
            result = predictor.predict(
                args.image_path,
                return_probabilities=True,
                confidence_threshold=args.threshold
            )
        
        print_prediction_result(result)
        
        # Save
        if args.save:
            import json
            output_path = f"{config['paths']['outputs']}/prediction_result.json"
            os.makedirs(config['paths']['outputs'], exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"✓ Result saved: {output_path}\n")
    
    except Exception as e:
        print(f"✗ Prediction error: {e}")


if __name__ == "__main__":
    main()