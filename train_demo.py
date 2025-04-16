import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from collections import Counter
import numpy as np

from models.enhanced_model import EnhancedCNNBiLSTM

# 데이터 로드
X_train = np.load("data3/split/X_train.npy")
y_train = np.load("data3/split/y_train.npy")
X_val = np.load("data3/split/X_val.npy")
y_val = np.load("data3/split/y_val.npy")

print("✅ 데이터 로드 완료")
print("Train 분포:", Counter(y_train.tolist()))
print("Val 분포:", Counter(y_val.tolist()))

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
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

# 모델 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedCNNEnsemble(num_classes=3, dropout=0.3).to(device)

# 손실함수 및 최적화
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 학습 루프 (간단히 10 epoch)
for epoch in range(10):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"[Epoch {epoch+1}] Loss: {total_loss/len(train_loader):.4f}")

# 평가
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        out = model(xb)
        preds = torch.argmax(out, dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(yb)

y_true = torch.cat(all_labels).numpy()
y_pred = torch.cat(all_preds).numpy()

print("\n📊 Validation 결과")
print(classification_report(y_true, y_pred, target_names=["Normal", "Crackle", "Wheeze"]))