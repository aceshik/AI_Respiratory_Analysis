import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score
import numpy as np
from collections import Counter
from tqdm import tqdm
from models.cnn_lstm_model import CustomCNNLSTM
from sklearn.utils import resample

# 데이터 로드
X_train = np.load("data2/X_train.npy")
y_train = np.load("data2/y_train.npy")
X_val = np.load("data2/X_val.npy")
y_val = np.load("data2/y_val.npy")

# 🔍 Wheeze 클래스만 오버샘플링
print("💡 Oversampling 이전 분포:", Counter(y_train))

target_count = 1100
X_aug, y_aug = [], []

for cls in [2]:  # Wheeze만 대상
    idx = np.where(y_train == cls)[0]
    samples_X = X_train[idx]
    samples_y = y_train[idx]

    if len(idx) < target_count:
        X_resampled, y_resampled = resample(samples_X, samples_y,
                                            replace=True,
                                            n_samples=target_count - len(idx),
                                            random_state=42)
        X_aug.append(X_resampled)
        y_aug.append(y_resampled)

X_train_balanced = np.concatenate([X_train] + X_aug, axis=0)
y_train_balanced = np.concatenate([y_train] + y_aug, axis=0)

print("✅ Oversampling 이후 분포:", Counter(y_train_balanced))

# 정규화
X_mean = X_train_balanced.mean()
X_std = X_train_balanced.std()
X_train = (X_train_balanced - X_mean) / X_std
X_val = (X_val - X_mean) / X_std

# Tensor로 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train_balanced, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

# DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

# 디바이스
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 정의 (클래스 수 = 3)
model = CustomCNNLSTM(num_classes=3).to(device)

# 손실함수 및 옵티마이저
ce_weights = torch.tensor([1.0, 1.0, 1.5]).to(device)
criterion = nn.CrossEntropyLoss(weight=ce_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# 평가 함수
def evaluate(model, dataloader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            preds.append(pred.cpu())
            labels.append(yb.cpu())

    y_true = torch.cat(labels).numpy()
    y_pred = torch.cat(preds).numpy()

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, labels=[0, 1, 2],
          target_names=["Normal", "Crackle", "Wheeze"], zero_division=0))
    print("▶ Predicted:", Counter(y_pred))
    print("▶ Ground Truth:", Counter(y_true))
    return f1_score(y_true, y_pred, average="macro")

# 학습 루프
for epoch in range(5):
    model.train()
    total_loss = 0

    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    val_f1 = evaluate(model, val_loader)
    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}")