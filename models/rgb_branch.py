import torch
import torch.nn as nn
from torchvision import models

class RGBBranch(nn.Module):
    def __init__(self, pretrained=True, frozen_epochs=0):
        super(RGBBranch, self).__init__()
        # Load EfficientNet-B0
        # weights='DEFAULT' corresponds to ImageNet
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Remove classification head
        # EfficientNet-B0 has 'classifier' as the last layer. 
        # The feature extractor part is 'features' + 'avgpool'.
        # However, `efficientnet_b0` forward returns the logits.
        # We want the 1280-D vector before the classifier.
        
        # Inspecting torchvision implementation:
        # self.classifier is a Sequential(Dropout, Linear)
        # The input to classifier is 1280.
        
        # We can just use the backbone but replace the classifier with Identity or just return features in forward.
        # Let's replace the classifier with Identity to be safe and standard.
        self.backbone.classifier = nn.Identity()
        
        # Feature dimension for B0 is 1280
        self.feature_dim = 1280

    def forward(self, x):
        # x: (B, 3, 224, 224)
        features = self.backbone(x)
        # features: (B, 1280)
        return features
    
    def set_trainable(self, trainable=True):
        for param in self.backbone.parameters():
            param.requires_grad = trainable
