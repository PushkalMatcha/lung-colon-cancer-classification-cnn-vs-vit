import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=5, pretrained=True):
    """Builds a ResNet50 model for transfer learning (Model 1 & 2)."""
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def get_vit_model(num_classes=5, pretrained=True):
    """Builds a Vision Transformer (ViT) model for transfer learning (Model 3 & 4)."""
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vit_b_16(weights=weights)

    for param in model.parameters():
        param.requires_grad = False
        
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, num_classes)
    
    return model