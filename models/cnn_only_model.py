import torch
import torch.nn as nn

class CNNOnly(nn.Module):
    def __init__(self, n_mfcc=13, time_steps=253, num_classes=2):
        super(CNNOnly, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 13, 253)
        x = self.features(x)  # (B, 64, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 64)
        return self.classifier(x)  # (B, num_classes)