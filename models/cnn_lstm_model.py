import torch
import torch.nn as nn

class CustomCNNLSTM(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomCNNLSTM, self).__init__()

        # CNN 부분: 채널 수 증가
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Pooling
        self.pool = nn.MaxPool2d((2, 2))  # 크기 절반 감소

        # LSTM 부분
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, 
                            batch_first=True, bidirectional=True)

        # FC 부분
        self.fc_block = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 입력: (B, 13, 253)
        x = x.unsqueeze(1)  # -> (B, 1, 13, 253)
        x = self.conv_block(x)  # -> (B, 256, 13, 253)
        x = self.pool(x)        # -> (B, 256, 6, 126) approx

        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # -> (B, sequence_len, 256)

        x, _ = self.lstm(x)  # -> (B, seq_len, 256)
        x = self.fc_block(x[:, -1])  # 마지막 타임스텝만 사용 -> (B, num_classes)

        return x
