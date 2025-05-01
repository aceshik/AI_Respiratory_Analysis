import torch
import torch.nn as nn

class RemasteredCNNBiLSTM(nn.Module):
    def __init__(self, num_classes = 3, dropout = 0.3):
        super(RemasteredCNNBiLSTM, self).__init__()

        self.concat_layer = nn.Conv1d(13, 128, kernel_size=1, stride=1)

        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=16, stride=1),   # 필터 연산, feature 추출
            nn.BatchNorm1d(64),                             # 정규화: 평균 0 분산 1
            nn.Dropout(dropout),                            # 일부 뉴런 랜덤 끄기
            nn.ReLU()                                       # 0보다 작으면 0, 아니면 그대로
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=16, stride=2),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=8, stride=1),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.lstm1 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.lstm2 = nn.LSTM(
            input_size=128 * 2,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x):
        x = self.concat_layer(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.permute(0,2,1)
        
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = x[:, -1, :]             # 전체 시퀸스 요약한 최종 벡터 선택

        x = self.fc1(x)
        x = self.fc2(x)

        return x
