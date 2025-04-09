import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
from sklearn.utils import resample
from itertools import product

from models.enhanced_model import EnhancedCNNBiLSTM

# Confusion Matrix 시각화
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Validation)")
    plt.show()

# Focal Loss 정의
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# 데이터 로드 및 오버샘플링
X_train = np.load("data2/X_train.npy")
y_train = np.load("data2/y_train.npy")
X_val = np.load("data2/X_val.npy")
y_val = np.load("data2/y_val.npy")

print("\n💡 Oversampling 이전 분포:", Counter(y_train))

# Wheeze 오버샘플링
wheeze_target = 1100
wheeze_idx = np.where(y_train == 2)[0]
X_wheeze = X_train[wheeze_idx]
y_wheeze = y_train[wheeze_idx]
X_resampled, y_resampled = resample(X_wheeze, y_wheeze, replace=True,
                                    n_samples=wheeze_target - len(y_wheeze), random_state=42)
X_train = np.concatenate([X_train, X_resampled])
y_train = np.concatenate([y_train, y_resampled])

print("✅ Oversampling 이후 분포:", Counter(y_train))

# 정규화
X_mean = X_train.mean()
X_std = X_train.std()
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std

# Tensor 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

# DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, drop_last=False)

# 디바이스 및 모델 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedCNNBiLSTM(num_classes=3).to(device)

# 손실함수 및 옵티마이저
# thresholds = [0.4, 0.4, 0.3]  # [Normal, Crackle, Wheeze]
ce_weights = torch.tensor([2.0, 2.0, 1.0]).to(device)
criterion = FocalLoss(alpha=ce_weights, gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# 평가 함수
def evaluate_with_threshold(model, dataloader, device, thresholds=None, verbose=True):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device).long().squeeze()
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)

            if thresholds is not None:
                pred = torch.full((probs.size(0),), -1, dtype=torch.long)
                for i, t in enumerate(thresholds):
                    mask = (probs[:, i] >= t)
                    pred[mask] = i
                fallback_mask = (pred == -1)
                pred[fallback_mask] = torch.argmax(probs[fallback_mask], dim=1)
            else:
                pred = torch.argmax(probs, dim=1)

            all_preds.append(pred.cpu())
            all_labels.append(yb.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    if verbose:
        print("\nFinal Evaluation:")
        print(classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=["Normal", "Crackle", "Wheeze"],
            zero_division=0
        ))
        print("▶ Predicted:", Counter(y_pred))
        print("▶ Ground truth:", Counter(y_true))

    return f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2])

# 학습 루프
for epoch in range(3):
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
        
    thresholds = [0.4, 0.4, 0.3]  # [Normal, Crackle, Wheeze]
    val_f1 = evaluate_with_threshold(model, val_loader, device, thresholds=thresholds)
    print(f"[Epoch {epoch+1}] Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

# Confusion Matrix
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        preds = torch.argmax(model(xb), dim=1).cpu()
        y_true.extend(yb.numpy())
        y_pred.extend(preds.numpy())

plot_confusion_matrix(y_true, y_pred, ["Normal", "Crackle", "Wheeze"])
