import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x, src_key_padding_mask=None):
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


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

class SimplifiedCNNLSTM(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        super(SimplifiedCNNLSTM, self).__init__()

        # 채널 수 맞추기
        self.concat_layer = nn.Conv1d(13, 128, kernel_size=1, stride=1)

        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=16, stride=1),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=16, stride=2),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.concat_layer(x)   # (B, 13, T) → (B, 128, T)
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.permute(0, 2, 1)     # (B, T, 128)

        x, _ = self.lstm(x)
        x = x[:, -1, :]            # 마지막 time step

        x = self.fc1(x)
        x = self.fc2(x)
        return x        

class Conv2dBiLSTM(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        super(Conv2dBiLSTM, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Conv2d 출력: (B, 128, 13, T) → LSTM 입력: (B, T, 128*13)
        self.lstm1 = nn.LSTM(
            input_size=128 * 13,
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

        self.fc_block = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 13, T) → (B, 1, 13, T)
        x = self.conv_block(x)  # → (B, 128, 13, T)

        x = x.permute(0, 3, 1, 2)  # → (B, T, 128, 13)
        x = x.reshape(x.size(0), x.size(1), -1)  # → (B, T, 128 * 13)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # 마지막 time step 사용

        x = self.fc_block(x)
        return x


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttentionBlock(nn.Module):
    def __init__(self, input_dim):
        super(AttentionBlock, self).__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (B, T, C)
        weights = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        weighted = x * weights  # (B, T, C)
        output = weighted.sum(dim=1)  # (B, C)
        return output, weights



class RemasteredCNNBiLSTMnoPadding(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        super(RemasteredCNNBiLSTMnoPadding, self).__init__()

        self.concat_layer = nn.Conv1d(13, 128, kernel_size=1, stride=1)

        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=16, stride=1, padding=8),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.positional_encoding = PositionalEncoding(256)

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
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(128, num_classes)

        self.attention = AttentionBlock(256)


    def forward(self, x, lengths):
        x = self.concat_layer(x)     # (B, 128, T)
        x = self.conv1(x)
        lengths = lengths + 2 * 8 - 16 + 1  # padding=8, kernel_size=16, stride=1
        x = self.conv2(x)
        lengths = (lengths + 2 * 8 - 16) // 2 + 1  # padding=8, kernel_size=16, stride=2
        x = self.conv3(x)
        lengths = lengths + 2 * 4 - 8 + 1  # padding=4, kernel_size=8, stride=1
        x = x.permute(0, 2, 1)       # (B, T', 256)
        x = self.positional_encoding(x)

        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm1(packed)
        out, output_lengths = pad_packed_sequence(packed_out, batch_first=True)

        packed = pack_padded_sequence(out, output_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm2(packed)
        out, output_lengths = pad_packed_sequence(packed_out, batch_first=True)

        x, _ = self.attention(out)
        #x = out[:, -1, :] 
        x = self.fc1(x)
        return self.fc2(x)


# New model with Transformer encoder after attention
class RemasteredCNNBiLSTMwithTransformer(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()

        self.concat_layer = nn.Conv1d(13, 128, kernel_size=1, stride=1)

        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=16, stride=1, padding=8),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.positional_encoding = PositionalEncoding(256)

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

        self.attention = AttentionBlock(256)

        self.transformer = SimpleTransformerEncoder(d_model=256, dropout=dropout)

        self.fc1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, lengths):
        x = self.concat_layer(x)
        x = self.conv1(x)
        lengths = lengths + 2 * 8 - 16 + 1
        x = self.conv2(x)
        lengths = (lengths + 2 * 8 - 16) // 2 + 1
        x = self.conv3(x)
        lengths = lengths + 2 * 4 - 8 + 1

        x = x.permute(0, 2, 1)
        x = self.positional_encoding(x)

        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm1(packed)
        out, output_lengths = pad_packed_sequence(packed_out, batch_first=True)

        packed = pack_padded_sequence(out, output_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm2(packed)
        out, output_lengths = pad_packed_sequence(packed_out, batch_first=True)

        x, _ = self.attention(out)
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        x = self.fc1(x)
        return self.fc2(x)


class MelSpectrogramCNN(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        super(MelSpectrogramCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, F, T)
        x = self.conv_block(x)
        return self.classifier(x)