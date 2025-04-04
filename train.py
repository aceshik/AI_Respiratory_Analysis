import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score
import numpy as np
from models.cnn_lstm_model import CNNLSTM

# 데이터 로드
X_train = np.load("data/processed/X_train_norm.npy")
y_train = np.load("data/processed/y_train.npy")
X_val = np.load("data/processed/X_val_norm.npy")
y_val = np.load("data/processed/y_val.npy")

# Tensor로 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# 클래스별 가중치 계산
pos_weight = (y_train.shape[0] - y_train.sum(dim=0)) / y_train.sum(dim=0)

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTM().to(device)

criterion_train = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
criterion_val = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

pos_weight = (y_train.shape[0] - y_train.sum(dim=0)) / y_train.sum(dim=0)
print("pos_weight:", pos_weight)

# 평가 함수 정의
def evaluate(model, dataloader, device, verbose=True):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            preds = torch.sigmoid(preds)  # 수동 시그모이드 적용
            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_pred_bin = (y_pred >= 0.5).astype(int)

    if verbose:
        print("\nFinal Evaluation:")
        print(classification_report(y_true, y_pred_bin, target_names=["Crackle", "Wheeze"]))

    return f1_score(y_true, y_pred_bin, average="macro")

    

# 학습 루프
for epoch in range(20):
    model.train()
    total_loss = 0
    num_batches = 0  # 추가

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion_train(preds, yb)

        optimizer.zero_grad()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name} grad mean: {param.grad.mean()}")  # 평균 gradient 출력
            else:
                print(f"{name} grad is None")  # gradient가 None인 경우 출력


        optimizer.step()

        total_loss += loss.item()
        num_batches += 1  # 배치 개수 세기

    # 평균 loss 출력으로 변경
    avg_train_loss = total_loss / num_batches

    
    # Validation loss
    model.eval()
    val_loss = 0
    num_val_batches = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion_val(preds, yb)  # 일반 loss 사용
            val_loss += loss.item()
            num_val_batches += 1

    avg_val_loss = val_loss / num_val_batches
    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# 마지막 Validation 결과 출력
_ = evaluate(model, val_loader, device, verbose=True)

# Test 데이터 로드 및 평가
X_test = np.load("data/processed/X_test_norm.npy")
y_test = np.load("data/processed/y_test.npy")

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# 최종 Test 평가
_ = evaluate(model, test_loader, device, verbose=True)