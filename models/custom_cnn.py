import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):  # crackle, wheeze
        super(CustomCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
            nn.ReLU()
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # (B, 256, 1, 1)
        self.fc1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 입력: (B, 13, 253)
        x = x.unsqueeze(1)  # (B, 1, 13, 253)

        x = self.conv1(x)  # (B, 64, 13, 253)
        x = self.conv2(x)  # (B, 128, 13, 253)
        x = self.conv3(x)  # (B, 256, 13, 253)

        x = self.global_pool(x)  # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)

        x = self.fc1(x)  # (B, 128)
        x = self.fc2(x)  # (B, 2)
        return x