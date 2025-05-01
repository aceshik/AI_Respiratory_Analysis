import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import pandas as pd

from models.remastered_model import RemasteredCNNBiLSTM

# ------------------ 기본 설정 ------------------
use_wheeze_aug = True
use_crackle_aug = True
dropout_rate = 0.35
batch_size = 32
epochs = 50
lr = 1e-3
thresholds_list = [0.44, 0.38, 0.42]  # Normal, Crackle, Wheeze
class_weights = [0.9, 2.5, 2.5]       # Normal, Crackle, Wheeze

# ------------------ 데이터 로딩 ------------------
X_train = np.load("data3/split/X_train.npy")
y_train = np.load("data3/split/y_train.npy")
X_val = np.load("data3/split/X_val.npy")
y_val = np.load("data3/split/y_val.npy")

# 정규화
X_mean = X_train.mean()
X_std = X_train.std()
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std

# 길이-비례 정규화
frame_lens_train = (X_train != 0).any(axis=1).sum(axis=1).reshape(-1, 1, 1)  # shape: (N, 1, 1)
frame_lens_val = (X_val != 0).any(axis=1).sum(axis=1).reshape(-1, 1, 1)

# 증강 데이터 병합 (metadata 기준으로 split == "train" 인 경우만 포함)
if use_wheeze_aug:
    meta = pd.read_csv("data3/mfcc_augmented/metadata_wheeze_aug.csv")
    train_idx = meta[meta["split"] == "train"].index
    Xw = np.load("data3/mfcc_augmented/X_wheeze_aug.npy")[train_idx]
    yw = np.load("data3/mfcc_augmented/y_wheeze_aug.npy")[train_idx]
    Xw = (Xw - X_mean) / X_std
    X_train = np.concatenate([X_train, Xw], axis=0)
    y_train = np.concatenate([y_train, yw], axis=0)

if use_crackle_aug:
    meta = pd.read_csv("data3/mfcc_augmented/metadata_crackle_aug.csv")
    train_idx = meta[meta["split"] == "train"].index
    Xc = np.load("data3/mfcc_augmented/X_crackle_aug.npy")[train_idx]
    yc = np.load("data3/mfcc_augmented/y_crackle_aug.npy")[train_idx]
    Xc = (Xc - X_mean) / X_std
    X_train = np.concatenate([X_train, Xc], axis=0)
    y_train = np.concatenate([y_train, yc], axis=0)

# Tensor 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

# DataLoader 정의
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

# ------------- 모델, 손실함수, 옵티마이저 ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RemasteredCNNBiLSTM(num_classes=3, dropout=dropout_rate).to(device)

# Force weight reinitialization
def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

model.apply(reset_weights)

## Class weights (no longer used with standard CrossEntropyLoss)
# label_counts = Counter(y_train.tolist())
# total = sum(label_counts.values())
# weights = torch.tensor([1.4, 1.9, 2.4], dtype=torch.float32).to(device)
# weights = weights / weights.mean()
# weights = weights.to(device)

weights = torch.tensor(class_weights, dtype=torch.float32).to(device)  # Normal, Crackle, Wheeze
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# EarlyStopping 클래스
class EarlyStopping:
    def __init__(self, patience=6):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, score, model):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_model = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ---------------- 학습 루프 ----------------------
train_f1s = []
val_f1s = []
train_losses = []
val_losses = []
early_stopper = EarlyStopping(patience=8)
best_val_f1 = 0.0

for epoch in range(epochs):
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

    train_losses.append(total_loss/len(train_loader))

    # 평가
    model.eval()
    y_true, y_pred = [], []
    val_loss = sum([criterion(model(xb.to(device)), yb.to(device)).item() for xb, yb in val_loader]) / len(val_loader)
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(yb.numpy())

    val_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"[Epoch {epoch+1}] Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

    val_f1s.append(val_f1)
    val_losses.append(val_loss)

    train_f1 = f1_score(y_train.numpy(), torch.argmax(model(X_train.to(device)).cpu(), dim=1), average='macro')
    train_f1s.append(train_f1)

    print(f"[Epoch {epoch+1}] Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
    scheduler.step(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "results/best_model.pth")
        print("Best model saved.")

    early_stopper(val_f1, model)
    if early_stopper.early_stop:
        print("Early stopping triggered.")
        model.load_state_dict(early_stopper.best_model)
        break



# Threshold-based prediction function and thresholds
def predict_with_threshold(logits, thresholds):
    probs = torch.softmax(logits, dim=1)
    preds = []
    for row in probs:
        pred = (row >= thresholds).int()
        if pred.sum() == 0:
            pred[torch.argmax(row)] = 1  # fallback to argmax if none pass threshold
        preds.append(torch.argmax(pred).item())
    return torch.tensor(preds)
thresholds = torch.tensor(thresholds_list).to(device)

print("\n🔍 수동 Threshold 적용 결과")
with torch.no_grad():
    logits = model(X_val.to(device))
    preds = predict_with_threshold(logits, thresholds)
print(f"Thresholds: {thresholds.tolist()}")
print(classification_report(y_val.cpu().numpy(), preds.cpu().numpy(), target_names=["Normal", "Crackle", "Wheeze"]))

plt.figure(figsize=(10, 4))
plt.plot(train_f1s, label="Train F1")
plt.plot(val_f1s, label="Val F1")
plt.plot(train_losses, 'r--', label="Train Loss")
plt.plot(val_losses, 'g--', label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Score / Loss")
plt.title("Training and Validation F1 Score and Loss")
plt.legend()
plt.grid(True)
plt.show()
