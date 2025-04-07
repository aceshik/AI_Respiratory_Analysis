import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 추가된 레이어
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 13, 253)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class EnhancedSimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(EnhancedSimpleCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.3),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(  # 추가된 레이어
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),
            nn.ReLU()
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # (B, 128, 1, 1)

        self.fc1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 13, 253)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # (B, 128)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
