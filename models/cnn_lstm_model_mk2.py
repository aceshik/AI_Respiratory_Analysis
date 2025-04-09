import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=4):  # crackle, wheeze, normal, c+w
        super(CNNLSTM, self).__init__()

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

        self.pool = nn.MaxPool2d((2, 2))  # 추가된 pooling layer

        self.lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)

        self.fc1 = nn.Sequential(
            nn.Linear(128 * 2, 128),
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

        x = self.pool(x)   # (B, 256, 6, 126) approx
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, sequence_len, 256)

        x, _ = self.lstm(x)  # (B, sequence_len, 256)
        x = self.fc1(x[:, -1])  # 마지막 시퀀스 (B, 256) → (B, 128)
        x = self.fc2(x)  # (B, 2)
        return x

