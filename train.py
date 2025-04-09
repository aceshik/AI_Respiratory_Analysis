import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score
import numpy as np
import matplotlib.pyplot as plt

from models.cnn_lstm_model import CNNLSTM

# 데이터 로드
X_train = np.load("data/augmentation/augmented/X_all_augmented_advanced.npy")
y_train = np.load("data/augmentation/augmented/y_all_augmented_advanced.npy")
X_val = np.load("data/processed/X_val_norm.npy")
y_val = np.load("data/processed/y_val.npy")

# Tensor로 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)  # ✅ long으로 변경
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)      # ✅ long으로 변경

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 정의
model = CNNLSTM().to(device)

# pos_weight 계산 (각 클래스별 양성 샘플 비율 반영)
# pos_weight = torch.tensor([2.5, 3.1]).to(device)  # ❌ CrossEntropyLoss에서는 불필요
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-5)

# 평가 함수 정의
from sklearn.metrics import classification_report, f1_score

def evaluate(model, dataloader, device, verbose=True):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device).long()
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)  # 가장 높은 확률의 클래스 선택
            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    if verbose:
        print("\nFinal Evaluation:")
        print(classification_report(
            y_true, y_pred,
            target_names=["Normal", "Crackle", "Wheeze", "Both"]
        ))

    return f1_score(y_true, y_pred, average="macro")

# 학습 루프
for epoch in range(1):  # 일단 1 epoch만
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device).long()  # ✅ yb long형 유지
        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    all_val_preds = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device).long()
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item()
            all_val_preds.append(torch.argmax(logits, dim=1).cpu())  # ✅ softmax, sigmoid 제거

    # F1
    val_f1 = evaluate(model, val_loader, device, verbose=True)

    print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | F1-score: {val_f1:.4f}")

    # 디버깅 출력
    sample_input = X_val[:8].to(device)
    model.eval()
    with torch.no_grad():
        logits = model(sample_input)
        preds = torch.argmax(logits, dim=1)
        print("\n=== Debug Info ===")
        print("Logits:\n", logits.cpu())
        print("Predicted classes:\n", preds.cpu())
        print("True labels:\n", y_val[:8])

    # 히스토그램
    plt.hist(torch.cat(all_val_preds).numpy().flatten(), bins=4)
    plt.title("Predicted Class Distribution (Validation)")
    plt.grid(True)
    plt.show()

# 마지막 평가
_ = evaluate(model, val_loader, device, verbose=True)