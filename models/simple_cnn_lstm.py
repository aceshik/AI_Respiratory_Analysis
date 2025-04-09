import torch
import torch.nn as nn

class SimpleCNNLSTM(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNNLSTM, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (B, 1, 13, 253) → (B, 16, 13, 253)
            nn.ReLU(),
            nn.MaxPool2d((2, 2))                         # → (B, 16, 6, 126)
        )

        self.lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(32 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 13, 253)
        x = self.conv(x)    # (B, 16, 6, 126)

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, sequence_len, features)

        x, _ = self.lstm(x)  # (B, seq_len, 64)
        x = x[:, -1]         # 마지막 시퀀스 벡터 사용

        out = self.fc(x)     # (B, num_classes)
        return out