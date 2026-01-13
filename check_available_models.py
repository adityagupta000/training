"""
Quick script to check available models in your timm installation
and test which ones work
"""

import timm
import torch

print("="*70)
print("TIMM MODEL AVAILABILITY CHECK")
print("="*70)
print(f"\ntimm version: {timm.__version__}")
print(f"torch version: {torch.__version__}\n")

# Models to test
test_models = [
    'efficientnet_b0',
    'efficientnet_b1', 
    'efficientnet_b2',
    'efficientnetv2_rw_s',
    'tf_efficientnet_b0',
    'tf_efficientnet_b2',
    'tf_efficientnetv2_b0',
    'tf_efficientnetv2_b2',
    'tf_efficientnetv2_b0.in1k',
    'tf_efficientnetv2_b2.in1k',
    'resnet50',
    'resnet50.a1_in1k',
    'convnext_tiny',
    'convnext_tiny.fb_in1k',
]

print("Testing specific models:")
print("-"*70)

working_models = []
failed_models = []

for model_name in test_models:
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=0)
        feature_dim = model.num_features
        print(f"✓ {model_name:40} → Feature dim: {feature_dim}")
        working_models.append(model_name)
        del model
    except Exception as e:
        print(f"✗ {model_name:40} → Failed: {str(e)[:30]}")
        failed_models.append(model_name)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n✓ Working models: {len(working_models)}")
print(f"✗ Failed models: {len(failed_models)}")

if working_models:
    print("\n" + "="*70)
    print("RECOMMENDED MODELS FOR YOUR CONFIG")
    print("="*70)
    
    # Prioritize EfficientNet models
    efficientnets = [m for m in working_models if 'efficient' in m.lower()]
    resnets = [m for m in working_models if 'resnet' in m.lower()]
    convnexts = [m for m in working_models if 'convnext' in m.lower()]
    
    if efficientnets:
        print("\nEfficientNet models (BEST for plant images):")
        for m in efficientnets[:5]:
            print(f"  ✓ {m}")
    
    if convnexts:
        print("\nConvNeXt models (GOOD for plant images):")
        for m in convnexts[:3]:
            print(f"  ✓ {m}")
    
    if resnets:
        print("\nResNet models (RELIABLE fallback):")
        for m in resnets[:3]:
            print(f"  ✓ {m}")
    
    print("\n" + "="*70)
    print("HOW TO UPDATE CONFIG.YAML")
    print("="*70)
    
    # Choose the best available model
    if 'efficientnet_b2' in working_models:
        recommended = 'efficientnet_b2'
    elif 'tf_efficientnetv2_b2.in1k' in working_models:
        recommended = 'tf_efficientnetv2_b2.in1k'
    elif any('efficient' in m for m in working_models):
        recommended = [m for m in working_models if 'efficient' in m][0]
    elif 'resnet50' in working_models:
        recommended = 'resnet50'
    else:
        recommended = working_models[0]
    
    print(f"\nRecommended: {recommended}")
    print("\nChange in config.yaml:")
    print("  model:")
    print(f"    name: '{recommended}'")
    print("    pretrained: true")
    print("    num_classes: 7")
    print("    dropout: 0.2")
    print("    unfreeze_layers: 35")

else:
    print("\n⚠ WARNING: No models could be loaded!")
    print("\nTry updating timm:")
    print("  pip install timm --upgrade")

print("\n" + "="*70)
print("\nSearching for all available EfficientNet models...")
print("-"*70)

try:
    all_efficient = timm.list_models('*efficient*', pretrained=True)
    print(f"\nFound {len(all_efficient)} EfficientNet variants:")
    for i, model in enumerate(all_efficient[:15]):
        print(f"  {i+1:2}. {model}")
    if len(all_efficient) > 15:
        print(f"  ... and {len(all_efficient)-15} more")
    
    print("\n✓ You can use ANY of these in config.yaml")
except Exception as e:
    print(f"Could not list models: {e}")

print("\n" + "="*70 + "\n")

# Test forward pass with recommended model
if working_models:
    print("Testing forward pass with recommended model...")
    try:
        test_model = timm.create_model(recommended, pretrained=False, num_classes=7)
        dummy_input = torch.randn(1, 3, 224, 224)
        output = test_model(dummy_input)
        print(f"✓ Forward pass successful! Output shape: {output.shape}")
        print(f"✓ {recommended} is ready to use!")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
    
    print("\n" + "="*70)
    print("READY TO TRAIN!")
    print("="*70)
    print(f"\n1. Update config.yaml with: name: '{recommended}'")
    print("2. Run: python train.py")
    print("\n" + "="*70 + "\n")