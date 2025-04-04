import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.2),  # Dropout 비율을 줄임
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.2)   # Dropout 비율을 줄임
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True, bidirectional=True)  # hidden_size 128로 설정
        self.fc = nn.Linear(128 * 2, 2)  # LSTM hidden_size * 2로 설정 (bidirectional)

        # 명시적 Weight 초기화 추가
        self.apply(self.init_weights)

    def forward(self, x):
        x = x.unsqueeze(1)  # 입력 shape (B, 13, 253) → (B, 1, 13, 253)
        x = self.cnn(x)
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # LSTM 입력에 맞게 차원 변경
        x, _ = self.lstm(x)  # LSTM 출력
        x = self.fc(x[:, -1])  # LSTM의 마지막 시퀀스를 출력으로
        return x

    # 수정된 weight 초기화 메소드
    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)