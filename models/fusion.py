import torch
import torch.nn as nn
from .rgb_branch import RGBBranch
from .fft_branch import FFTBranch

class DeepFakeDetector(nn.Module):
    def __init__(self, config):
        super(DeepFakeDetector, self).__init__()
        
        # Initialize branches
        self.rgb_branch = RGBBranch(
            pretrained=config['model']['pretrained']
        )
        
        self.fft_branch = FFTBranch(
            feature_dim=config['model']['fft_feature_dim']
        )
        
        # Fusion Head
        rgb_dim = self.rgb_branch.feature_dim
        fft_dim = self.fft_branch.feature_dim
        fusion_input_dim = rgb_dim + fft_dim # 1280 + 512 = 1792
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout_rate']),
            nn.Linear(512, 1)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb_input, fft_input):
        # rgb_input: (B, 3, 224, 224)
        # fft_input: (B, 1, 224, 224)
        
        rgb_features = self.rgb_branch(rgb_input)
        fft_features = self.fft_branch(fft_input)
        
        # Fusion
        combined = torch.cat((rgb_features, fft_features), dim=1)
        
        # Classification
        logits = self.classifier(combined)
        probs = self.sigmoid(logits)
        
        return logits, probs
