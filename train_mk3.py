import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score

from models.enhanced_model import EnhancedCNNBiLSTM, EnhancedCNNEnsemble

# 데이터 로드
X_train = np.load("data2/X_train.npy")
y_train = np.load("data2/y_train.npy")
X_val = np.load("data2/X_val.npy")
y_val = np.load("data2/y_val.npy")

print("💡 정규화 이전 분포:", Counter(y_train))

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

# 디바이스 설정 및 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedCNNEnsemble(num_classes=3).to(device)

# FocalLoss 클래스 정의
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-CE_loss)  # Probability of correct class
        F_loss = (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# 손실 함수 및 옵티마이저
ce_weights = torch.tensor([0.995, 1.118, 0.908]).to(device)
criterion = FocalLoss(alpha=ce_weights, gamma=1.5)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 평가 함수
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    print(classification_report(y_true, y_pred, target_names=["Normal", "Crackle", "Wheeze"], zero_division=0))
    print("▶ Predicted:", Counter(y_pred))
    print("▶ Ground truth:", Counter(y_true))

    return f1_score(y_true, y_pred, average="macro")

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

    val_f1 = evaluate(model, val_loader, device)
    print(f"[Epoch {epoch+1}] Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")
