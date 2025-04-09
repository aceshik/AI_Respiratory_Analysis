import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score
import numpy as np
from models.simple_cnn import EnhancedSimpleCNN  # 모델 변경

# 데이터 로드
X_train = np.load("data/processed/X_train_norm.npy")
y_train = np.load("data/processed/y_train.npy")
X_val = np.load("data/processed/X_val_norm.npy")
y_val = np.load("data/processed/y_val.npy")

# Tensor 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

# 모델, 장비 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedSimpleCNN().to(device)

# pos_weight 계산
num_pos = y_train.sum(dim=0)
num_neg = y_train.shape[0] - num_pos
pos_weight = (num_neg / num_pos).to(device)

# Loss & Optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 평가 함수
def evaluate(model, dataloader, threshold=0.4, verbose=True):
    model.eval()
    preds_list, labels_list = [], []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = torch.sigmoid(model(xb))
            preds_list.append(outputs.cpu())
            labels_list.append(yb.cpu())

    y_pred = torch.cat(preds_list).numpy()
    y_true = torch.cat(labels_list).numpy()
    y_bin = (y_pred >= threshold).astype(int)

    if verbose:
        print("\nFinal Evaluation:")
        print(classification_report(y_true, y_bin, target_names=["Crackle", "Wheeze"]))
    return f1_score(y_true, y_bin, average="macro")

# 학습 루프
for epoch in range(20):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        outputs = model(xb)
        loss = criterion(outputs, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    
    # 검증
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += criterion(model(xb), yb).item()
    avg_val_loss = val_loss / len(val_loader)

    val_f1 = evaluate(model, val_loader, threshold=0.4, verbose=False)

    print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} (avg {avg_loss:.4f}) | "
          f"Val Loss: {val_loss:.4f} (avg {avg_val_loss:.4f}) | F1-score: {val_f1:.4f}")

# 최종 평가
_ = evaluate(model, val_loader, threshold=0.4, verbose=True)