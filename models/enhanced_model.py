import torch
import torch.nn as nn

class EnhancedCNNBiLSTM(nn.Module):
    def __init__(self, num_classes=3):
        super(EnhancedCNNBiLSTM, self).__init__()

        # CNN block
        self.cnn_block = nn.Sequential(
            nn.Conv1d(in_channels=13, out_channels=128, kernel_size=5, padding=2),  # 채널 증가
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2),  # 채널 증가
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, padding=2),  # 채널 증가
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # BiLSTM Layer
        self.lstm = nn.LSTM(input_size=512, hidden_size=128,  # hidden_size 증가
                            num_layers=3, batch_first=True,  # 레이어 수 증가
                            dropout=0.3, bidirectional=True)

        # Fully Connected block
        self.fc_block = nn.Sequential(
            nn.Linear(128 * 2, 512),  # hidden_size 증가, 더 많은 뉴런
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: (B, 13, 253)
        x = self.cnn_block(x)  # (B, 256, T/4)
        x = x.permute(0, 2, 1)  # (B, T/4, 256)

        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)  # (B, T/4, 256)
        x_last = lstm_out[:, -1, :]  # 마지막 타임스텝 (B, 256)
        out = self.fc_block(x_last)  # (B, num_classes)
        return out


class EnhancedCNNEnsemble(nn.Module):
    def __init__(self, num_classes=3):
        super(EnhancedCNNEnsemble, self).__init__()

        # Concatenate input size 128x3
        self.concat_layer = nn.Conv1d(13, 128, kernel_size=1, stride=1)

        # Conv1 Layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=32, stride=2),  # 128x3 -> 64x32
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.ReLU()
        )

        # Conv2 Layer
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=32, stride=2),  # 64x32 -> 128x16
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.ReLU()
        )

        # Conv3 Layer
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=32, stride=1),  # 128x16 -> 256x16
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.ReLU()
        )

        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Output size will be 256x1

        # LSTM       
        self.lstm = nn.LSTM(
            input_size=256, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.3, 
            bidirectional=True
        )


        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Concatenate input
        x = self.concat_layer(x)  # (B, 128, 3)

        # Pass through Conv1
        x = self.conv1(x)  # (B, 64, 32)

        # Pass through Conv2
        x = self.conv2(x)  # (B, 128, 16)

        # Pass through Conv3
        x = self.conv3(x)  # (B, 256, 16)

        # Global Pooling (Reduce the sequence length to 1)
        x = self.global_pool(x)  # (B, 256, 1)
        x = x.squeeze(-1)  # (B, 256)

        # Fully Connected Layer 1
        x = self.fc1(x)  # (B, 128)

        # Fully Connected Layer 2 (Output Layer)
        x = self.fc2(x)  # (B, num_classes)

        return x