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
import os
import random


from models.remastered_model import RemasteredCNNBiLSTMnoPadding, SimplifiedCNNLSTM

# ------------------ 기본 설정 ------------------
model_name = RemasteredCNNBiLSTMnoPadding
use_wheeze_aug = True
use_crackle_aug = True
dropout_rate = 0.3
batch_size = 32
epochs = 50
lr = 1e-4
thresholds_list = [0.28, 0.48, 0.42]
class_weights = [1.8, 2.1, 2.3]
gamma_list = [1.5, 1.3, 1.8]

# 시각화 설정
show_graph = True
show_mfcc_grid = True
show_length_hist = True


# ------------------ 사전 정의 -------------------
# 클래스별 gamma
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=gamma_list, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-CE_loss)
        gamma_tensor = torch.tensor([self.gamma[t.item()] for t in targets]).to(inputs.device)
        focal_loss = ((1 - pt) ** gamma_tensor) * CE_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

def collate_fn(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.size(0) for x in xs])
    xs_padded = pad_sequence(xs, batch_first=True)  # shape: (B, T, 13)
    xs_padded = xs_padded.transpose(1, 2)           # transpose to (B, 13, T)
    ys = torch.tensor(ys)
    return xs_padded, lengths, ys




# ------------------ 데이터 로딩 ------------------
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

X_train = torch.load("data3/nopadding/X_train.pt")
y_train = torch.load("data3/nopadding/y_train.pt")
X_val = torch.load("data3/nopadding/X_val.pt")
y_val = torch.load("data3/nopadding/y_val.pt")

train_dataset = list(zip(X_train, y_train))
val_dataset = list(zip(X_val, y_val))

# Wheeze augmentation 합치기
if use_wheeze_aug:
    aug_x_path = "data3/nopadding/X_wheeze_aug_train.pt"
    aug_y_path = "data3/nopadding/y_wheeze_aug_train.pt"

    if os.path.exists(aug_x_path) and os.path.exists(aug_y_path):
        aug_X = torch.load(aug_x_path)
        aug_y = torch.tensor(torch.load(aug_y_path))

        # 원하는 수만큼 랜덤 선택
        target_n = 450
        selected_indices = random.sample(range(len(aug_X)), target_n)
        aug_X = [aug_X[i] for i in selected_indices]
        aug_y = [int(aug_y[i]) for i in selected_indices]

        # 기존 train 데이터 병합
        train_X, train_y = zip(*train_dataset)
        combined_X = list(train_X) + aug_X
        combined_y = list(map(int, train_y)) + aug_y
        train_dataset = list(zip(combined_X, combined_y))

        print(f"[INFO] Loaded and merged {target_n} Wheeze augmented samples.")
    else:
        print("[INFO] Wheeze augmentation enabled but no data found.")

# Crackle augmentation 합치기
if use_crackle_aug:
    aug_x_path = "data3/nopadding/X_crackle_noise_aug_train.pt"
    aug_y_path = "data3/nopadding/y_crackle_noise_aug_train.pt"

    if os.path.exists(aug_x_path) and os.path.exists(aug_y_path):
        aug_X = torch.load(aug_x_path)
        aug_y = torch.tensor(torch.load(aug_y_path))

        # 원하는 수만큼 랜덤 선택
        target_n = 180
        selected_indices = random.sample(range(len(aug_X)), target_n)
        aug_X = [aug_X[i] for i in selected_indices]
        aug_y = [int(aug_y[i]) for i in selected_indices]

        # 기존 train 데이터 병합
        train_X, train_y = zip(*train_dataset)
        combined_X = list(train_X) + aug_X
        combined_y = list(map(int, train_y)) + aug_y
        train_dataset = list(zip(combined_X, combined_y))

        print(f"[INFO] Loaded and merged {target_n} Crackle noise-augmented samples.")
    else:
        print("[INFO] Crackle augmentation enabled but no data found.")


    aug_x_path2 = "data3/nopadding/X_crackle_pitch_aug_train.pt"
    aug_y_path2 = "data3/nopadding/y_crackle_pitch_aug_train.pt"

    if os.path.exists(aug_x_path2) and os.path.exists(aug_y_path2):
        aug_X2 = torch.load(aug_x_path2)
        aug_y2 = torch.tensor(torch.load(aug_y_path2))

        target_n2 = 180
        selected_indices2 = random.sample(range(len(aug_X2)), min(target_n2, len(aug_X2)))
        aug_X2 = [aug_X2[i] for i in selected_indices2]
        aug_y2 = [int(aug_y2[i]) for i in selected_indices2]

        # 기존 train 데이터 병합
        train_X, train_y = zip(*train_dataset)
        combined_X = list(train_X) + aug_X2
        combined_y = list(map(int, train_y)) + aug_y2
        # Ensure all input tensors are of shape (T, 13)
        combined_X = [x.T if x.shape[0] == 13 else x for x in combined_X]
        train_dataset = list(zip(combined_X, combined_y))

        print(f"[INFO] Loaded and merged {target_n2} Crackle pitch-augmented samples.")
    else:
        print("[INFO] Crackle pitch augmentation enabled but no data found.")


# Normal 언더샘플링
from collections import defaultdict

class_buckets = defaultdict(list)
for x, y in train_dataset:
    class_buckets[y].append((x, y))

normal_limit = 600
reduced_train_dataset = (
    random.sample(class_buckets[0], min(normal_limit, len(class_buckets[0]))) +
    class_buckets[1] +
    class_buckets[2]
)
random.shuffle(reduced_train_dataset)
train_dataset = reduced_train_dataset

# ----- 전체 데이터 병합 후 재분할 -----
full_dataset = train_dataset + val_dataset
X_all, y_all = zip(*full_dataset)

from sklearn.model_selection import train_test_split

X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(
    X_all, y_all,
    test_size=0.15,
    stratify=y_all,
    random_state=42
)

train_dataset = list(zip(X_train_new, y_train_new))
val_dataset = list(zip(X_val_new, y_val_new))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

from collections import Counter
train_labels = [y for _, y in train_dataset]
val_labels = [y for _, y in val_dataset]
print(f"[Train Set Class Distribution] {dict(Counter(train_labels))}")
print(f"[Val Set Class Distribution] {dict(Counter(val_labels))}")


# ------------- 모델, 손실함수, 옵티마이저 ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.remastered_model import RemasteredCNNBiLSTMnoPadding

model = model_name(num_classes=3, dropout=dropout_rate).to(device)

# Force weight reinitialization
def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

model.apply(reset_weights)

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
train_accs = []
val_accs = []
early_stopper = EarlyStopping(patience=6)
best_val_f1 = 0.0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, lengths, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        xb, lengths, yb = xb.to(device), lengths.to(device), yb.to(device)
        logits = model(xb, lengths)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss/len(train_loader))

    # 평가
    model.eval()
    with torch.no_grad():
        y_true, y_pred = [], []
        val_loss = 0
        for xb, lengths, yb in val_loader:
            xb, lengths, yb = xb.to(device), lengths.to(device), yb.to(device)
            logits = model(xb, lengths)
            val_loss += criterion(logits, yb).item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(yb.cpu().numpy())
    val_loss /= len(val_loader)

    val_f1 = f1_score(y_true, y_pred, average='macro')
    val_acc = (np.array(y_true) == np.array(y_pred)).mean()
    print(f"[Epoch {epoch+1}] Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

    val_f1s.append(val_f1)
    val_losses.append(val_loss)

    # Compute train F1 on full training dataset
    model.eval()
    train_preds = []
    train_labels = []
    with torch.no_grad():
        for xb, lengths, yb in train_loader:
            xb, lengths, yb = xb.to(device), lengths.to(device), yb.to(device)
            logits = model(xb, lengths)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(yb.cpu().numpy())
    train_f1 = f1_score(train_labels, train_preds, average='macro')
    train_acc = (np.array(train_labels) == np.array(train_preds)).mean()
    train_f1s.append(train_f1)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"[Epoch {epoch+1}] Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    scheduler.step(val_f1)

    # Best Model 저장(Val F1 기준)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1,
            'train_f1': train_f1,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_loss': train_losses[-1],
            'val_loss': val_loss,
            'thresholds': thresholds_list,
            'class_weights': class_weights,
            'gamma_list': gamma_list
        }, f"results/best_model_f1_{val_f1:.4f}.pth")
        import os
        best_model_path = f"results/best_model_f1_{val_f1:.4f}.pth"
        symlink_path = "results/best_model.pth"

        if os.path.exists(symlink_path) or os.path.islink(symlink_path):
            try:
                os.unlink(symlink_path)
            except Exception:
                os.remove(symlink_path)
        os.symlink(best_model_path, symlink_path)
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
model.eval()
with torch.no_grad():
    all_logits = []
    for xb, lengths, yb in val_loader:
        xb, lengths = xb.to(device), lengths.to(device)
        logits = model(xb, lengths)
        all_logits.append(logits)
    logits = torch.cat(all_logits, dim=0)
    preds = predict_with_threshold(logits, thresholds)
print(f"Thresholds: {thresholds.tolist()}")
val_labels = [y for _, y in val_dataset]
print(classification_report(val_labels, preds.cpu().numpy(), target_names=["Normal", "Crackle", "Wheeze"]))



# -------------- 시각화: 학습 그래프 ----------------
if show_graph:
    plt.figure(figsize=(10, 5))
    plt.plot(train_f1s, label="Train F1")
    plt.plot(val_f1s, label="Val F1")
    plt.plot(train_losses, 'r--', label="Train Loss")
    plt.plot(val_losses, 'g--', label="Val Loss")
    plt.plot(train_accs, 'b-.', label="Train Acc")
    plt.plot(val_accs, 'm-.', label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Score / Loss / Acc")
    plt.title("Training and Validation Metrics")
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------- 시각화: 샘플 길이 분포 확인 ---------------
if show_length_hist:
    lengths = [x.shape[1] if x.shape[0] == 13 else x.shape[0] for x, _ in train_dataset]
    plt.hist(lengths, bins=30)
    plt.title("Length distribution of training samples")
    plt.xlabel("Sequence length (T)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


# ---------------- 시각화: 클래스별 MFCC(25개씩) ----------------
if show_mfcc_grid:
    def plot_mfcc_grid(samples, title):
        fig, axes = plt.subplots(5, 5, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)
        for idx, (x, _) in enumerate(samples[:25]):
            ax = axes[idx // 5, idx % 5]
            mfcc = x.transpose(0, 1).numpy()  # (T, 13) 형태로 변환
            ax.imshow(mfcc.T, aspect='auto', origin='lower', cmap='viridis')
            ax.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

    # 클래스별 샘플 분리
    from collections import defaultdict
    class_samples = defaultdict(list)
    for x, y in train_dataset:
        class_samples[y].append((x, y))

    # 각 클래스별 25개 시각화
    plot_mfcc_grid(class_samples[0][:25], "Normal (Class 0)")
    plot_mfcc_grid(class_samples[1][:25], "Crackle (Class 1)")
    plot_mfcc_grid(class_samples[2][:25], "Wheeze (Class 2)")
