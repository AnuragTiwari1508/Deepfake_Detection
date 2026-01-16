import torch
import torch.nn as nn

class FFTBranch(nn.Module):
    def __init__(self, feature_dim=512):
        super(FFTBranch, self).__init__()
        
        # Input: 224x224x1
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 112x112
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 56x56
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Global Avg Pool -> 1x1x128
        )
        
        self.projector = nn.Linear(128, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x):
        # x: (B, 1, 224, 224)
        x = self.features(x)
        x = torch.flatten(x, 1) # (B, 128)
        x = self.projector(x)   # (B, 512)
        return x
