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

from models.enhanced_model import EnhancedCNNBiLSTM, EnhancedCNNEnsemble, EnhancedCNNDeepBiLSTM

# 데이터 로드
X_train = np.load("data2/X_train.npy")
y_train = np.load("data2/y_train.npy")
X_val = np.load("data2/X_val.npy")
y_val = np.load("data2/y_val.npy")

print("정규화 이전 분포:", Counter(y_train))

# 정규화
X_mean = X_train.mean()
X_std = X_train.std()
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std

np.save("data2/X_mean_std.npy", [X_mean, X_std])

# Tensor 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

# Wheeze 증강 데이터 추가 (Pitch Shifting)
X_wheeze_ps = np.load("data2/X_wheeze_pitch_padded.npy")
y_wheeze_ps = np.load("data2/y_wheeze_pitch_padded.npy")

print(f"Wheeze 증강 데이터: {X_wheeze_ps.shape}, {Counter(y_wheeze_ps.tolist())}")

# (B, 200, 13) → (B, 13, 200) for consistency
X_wheeze_ps = np.transpose(X_wheeze_ps, (0, 2, 1))

# 정규화
X_wheeze_ps = (X_wheeze_ps - X_mean) / X_std

# Tensor 변환
X_wheeze_ps = torch.tensor(X_wheeze_ps, dtype=torch.float32)
y_wheeze_ps = torch.tensor(y_wheeze_ps, dtype=torch.long)

# concat
X_train = torch.cat([X_train, X_wheeze_ps], dim=0)
y_train = torch.cat([y_train, y_wheeze_ps], dim=0)

print("증강 데이터 포함 후 분포:", Counter(y_train.tolist()))

# 데이터 증강!!!
# 1. 원본 보존
original_X_train = X_train.clone()
original_y_train = y_train.clone()

# 2. 타겟 개수 설정 (각 클래스 개수 맞추기)
target_per_class = {
    0: 1400,  # Normal
    1: 1400,  # Crackle
    2: 1400   # Wheeze
}

# 3. 클래스별 인덱스 추출
normal_idx   = (original_y_train == 0).nonzero(as_tuple=True)[0]
crackle_idx  = (original_y_train == 1).nonzero(as_tuple=True)[0]
wheeze_idx   = (original_y_train == 2).nonzero(as_tuple=True)[0]

# 4. 클래스별 데이터 추출
X_normal,  y_normal  = original_X_train[normal_idx],  original_y_train[normal_idx]
X_crackle, y_crackle = original_X_train[crackle_idx], original_y_train[crackle_idx]
X_wheeze,  y_wheeze  = original_X_train[wheeze_idx],  original_y_train[wheeze_idx]

# 5. 증강 함수
def augment_to_target(X_class, y_class, target_count):
    current = len(X_class)
    if current >= target_count:
        return X_class[:target_count], y_class[:target_count]
    repeat_factor = target_count // current
    remainder = target_count % current
    X_aug = X_class.repeat((repeat_factor, 1, 1))
    y_aug = y_class.repeat(repeat_factor)
    if remainder > 0:
        X_aug = torch.cat([X_aug, X_class[:remainder]], dim=0)
        y_aug = torch.cat([y_aug, y_class[:remainder]], dim=0)
    return X_aug, y_aug

# 복사본에만 노이즈 추가 함수
def augment_to_target_with_noise(X_class, y_class, target_count, noise_std=0.05):
    current = len(X_class)
    if current >= target_count:
        return X_class[:target_count], y_class[:target_count]

    repeat_factor = target_count // current
    remainder = target_count % current

    # 복제
    X_aug = X_class.repeat((repeat_factor, 1, 1))
    y_aug = y_class.repeat(repeat_factor)

    # Gaussian noise 추가
    noise = torch.randn_like(X_aug) * noise_std
    X_aug = X_aug + noise

    # 나머지 샘플도 추가
    if remainder > 0:
        X_extra = X_class[:remainder]
        noise_extra = torch.randn_like(X_extra) * noise_std
        X_extra = X_extra + noise_extra

        X_aug = torch.cat([X_aug, X_extra], dim=0)
        y_aug = torch.cat([y_aug, y_class[:remainder]], dim=0)

    return X_aug, y_aug

# 복사본에만 timeshift 추가 함수
def time_shift(X_class, shift_max=10):
    B, C, T = X_class.shape
    shifted = torch.zeros_like(X_class)
    for i in range(B):
        shift = np.random.randint(-shift_max, shift_max + 1)
        if shift > 0:
            shifted[i, :, shift:] = X_class[i, :, :-shift]
        elif shift < 0:
            shifted[i, :, :shift] = X_class[i, :, -shift:]
        else:
            shifted[i] = X_class[i]
    return shifted
def augment_to_target_with_shift(X_class, y_class, target_count, shift_max=10):
    current = len(X_class)
    if current >= target_count:
        return X_class[:target_count], y_class[:target_count]

    repeat_factor = target_count // current
    remainder = target_count % current

    X_aug = X_class.repeat((repeat_factor, 1, 1))
    y_aug = y_class.repeat(repeat_factor)

    # 시간 이동 적용
    X_aug = time_shift(X_aug, shift_max=shift_max)

    if remainder > 0:
        X_extra = time_shift(X_class[:remainder], shift_max=shift_max)
        X_aug = torch.cat([X_aug, X_extra], dim=0)
        y_aug = torch.cat([y_aug, y_class[:remainder]], dim=0)

    return X_aug, y_aug    



# 6. 증강 수행
X_normal_aug,  y_normal_aug  = augment_to_target(X_normal,  y_normal,  target_per_class[0])
X_crackle_aug, y_crackle_aug = augment_to_target(X_crackle, y_crackle, target_per_class[1])
X_wheeze_aug,  y_wheeze_aug  = augment_to_target(X_wheeze,  y_wheeze,  target_per_class[2])

# 7. 최종 학습 데이터 구성
X_train = torch.cat([X_normal_aug, X_crackle_aug, X_wheeze_aug], dim=0)
y_train = torch.cat([y_normal_aug, y_crackle_aug, y_wheeze_aug], dim=0)

print("✅ 증강 완료:", Counter(y_train.tolist()))


# DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, drop_last=False)

# 디바이스 설정 및 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dropout = 0.3
model = EnhancedCNNEnsemble(num_classes=3, dropout=current_dropout).to(device)

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

# EarlyStopping 클래스 정의
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_score, model):
        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.best_model = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"조기 종료 카운트: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = model.state_dict()
            self.counter = 0


# 손실 함수 및 옵티마이저
total = sum(target_per_class.values())
ce_weights = torch.tensor([
    total / target_per_class[0],
    total / target_per_class[1],
    total / target_per_class[2]
])
ce_weights = ce_weights / ce_weights.mean()  # 정규화
ce_weights = ce_weights.to(device)

criterion = FocalLoss(alpha=ce_weights, gamma=1.0)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# 성능이 정체되면 학습률을 0.5배로 감소
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# threshold 기반 예측 함수
def predict_with_threshold(logits, thresholds):
    probs = torch.softmax(logits, dim=1)
    preds = torch.zeros(len(probs), dtype=torch.long)
    
    for i, p in enumerate(probs):
        over_th = (p >= thresholds).nonzero(as_tuple=True)[0]
        if len(over_th) > 0:
            preds[i] = over_th[torch.argmax(p[over_th])]
        else:
            preds[i] = torch.argmax(p)
    return preds

# threshold 탐색 함수
def find_best_thresholds(model, dataloader, device):
    model.eval()
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu())
            all_labels.append(yb)
    
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    
    best_f1 = 0
    best_thresholds = torch.tensor([0.5, 0.5, 0.5])  # 초기값
    
    for t0 in np.arange(0.1, 0.9, 0.05):
        for t1 in np.arange(0.1, 0.9, 0.05):
            for t2 in np.arange(0.1, 0.9, 0.05):
                thresholds = torch.tensor([t0, t1, t2])
                preds = predict_with_threshold(logits, thresholds)
                f1 = f1_score(labels, preds, average="macro")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresholds = thresholds
    
    print(f"🔥 Best Thresholds: {best_thresholds.tolist()}, Best F1: {best_f1:.4f}")
    return best_thresholds

# val accuracy 계산 함수
def calculate_val_accuracy(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)
    return correct / total

# 평가 함수
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    print(classification_report(y_true, y_pred, target_names=["Normal", "Crackle", "Wheeze"], zero_division=0))
    print("▶ Predicted:", Counter(y_pred))
    print("▶ Ground truth:", Counter(y_true))

    avg_loss = total_loss / len(dataloader)
    return f1_score(y_true, y_pred, average="macro"), avg_loss

# 성능 기록 리스트
train_f1_scores = []
val_f1_scores = []
train_accuracies = []
val_accuracies = []

# dropout 변경 시점 기록용 리스트
dropout_change_epochs = []
dropout_changed = False

# EarlyStopping 초기화
early_stopper = EarlyStopping(patience=6, verbose=True)

# best_val 초기화
best_val_f1 = 0.0

# 학습 루프
for epoch in range(30):
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 정확도 계산 (train 데이터)
        _, preds = torch.max(logits, 1)
        correct_train += torch.sum(preds == yb)
        total_train += yb.size(0)

    # 학습 후 평가
    train_f1, _ = evaluate(model, train_loader, device)
    val_f1, val_loss = evaluate(model, val_loader, device)

    # 현재 학습률 확인
    current_lr = optimizer.param_groups[0]['lr']
    print(f"[Epoch {epoch+1}] Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.6f}")
    
    # 스케줄러 동작
    scheduler.step(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "results/best_model.pth")
        print(f"Best model saved at Epoch {epoch+1} (Val F1: {val_f1:.4f})")

    # EarlyStopping 체크 및 Dropout 조정
    if early_stopper.counter >= 3 and not dropout_changed:
        current_dropout = 0.2
        dropout_changed_epoch = epoch + 1
        dropout_change_epochs.append(dropout_changed_epoch)
        dropout_changed = True

        print(f"🔁 Early stop 카운터 {early_stopper.counter} → Dropout을 {current_dropout}로 변경")
        
        # 모델 재생성 및 가중치 이전
        model = EnhancedCNNEnsemble(num_classes=3, dropout=current_dropout).to(device)
        model.load_state_dict(early_stopper.best_model)

        # Optimizer 재설정
        optimizer = optim.Adam(model.parameters(), lr=current_lr, weight_decay=1e-5)

    # 성능 기록
    train_f1_scores.append(train_f1)
    val_f1_scores.append(val_f1)
    
    # 정확도 계산
    train_accuracy = correct_train.item() / total_train
    val_accuracy = calculate_val_accuracy(model, val_loader, device)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"[Epoch {epoch+1}] Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Train Accuracy: {train_accuracy:.4f} | Val Accuracy: {val_accuracy:.4f}")

    
    # 얼리스탑
    early_stopper(val_f1, model)

    if early_stopper.early_stop:
        print("조기 종료 조건 만족. 학습을 종료합니다.")
        model.load_state_dict(early_stopper.best_model)
        break


#best_thresholds = find_best_thresholds(model, val_loader, device)
manual_thresholds = torch.tensor([0.5, 0.45, 0.4])

# 적용해서 예측
model.eval()
with torch.no_grad():
    logits = model(X_val.to(device))
    preds = predict_with_threshold(logits, manual_thresholds)

# 평가
print(classification_report(y_val.numpy(), preds.numpy(), target_names=["Normal", "Crackle", "Wheeze"]))


# 정보들
print("\n📌 Training Completed.")
print(f"Final Thresholds: {manual_thresholds.tolist()}")
print(f"Class Weights (ce_weights): {ce_weights.tolist()}")
print(f"Focal Loss Gamma: {criterion.gamma}")

# 성능 시각화 (F1-score & Accuracy)
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_f1_scores) + 1), train_f1_scores, label="Train F1-score", marker='o')
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy", marker='o')
plt.plot(range(1, len(val_f1_scores) + 1), val_f1_scores, label="Validation F1-score", marker='o')

# Dropout 조정 시점 표시
if 'dropout_changed_epoch' in locals():
    for i, change_epoch in enumerate(dropout_change_epochs):
        plt.axvline(x=change_epoch, color='orange', linestyle='--',
                    label=f'Dropout changed (Epoch {change_epoch})' if i == 0 else None)

plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Training & Validation F1-score / Accuracy')
plt.legend()
plt.show()

'''
# Feature Map 시각화
def visualize_feature_maps(model, input_tensor, device, title_prefix=""):
    model.eval()
    input_tensor = input_tensor.unsqueeze(0).to(device)  # (1, 13, 253)
    
    activations = {}

    def hook_fn(module, input, output):
        activations['conv1'] = output.cpu()

    # Conv1d만 hook
    handle = model.conv1[0].register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(input_tensor)

    handle.remove()

    feature_maps = activations['conv1'][0]  # shape: (C, T)

    num_maps = min(8, feature_maps.shape[0])
    plt.figure(figsize=(15, 5))
    for i in range(num_maps):
        plt.subplot(1, num_maps, i + 1)
        plt.imshow(feature_maps[i].unsqueeze(0), aspect='auto', origin='lower', cmap='viridis')
        plt.title(f"Filter {i}")
        plt.axis('off')
    plt.suptitle(f"{title_prefix} - Feature Maps from Conv1")
    plt.tight_layout()
    plt.show()

# 각 클래스별 대표 sample 추출
for label, name in zip([0, 1, 2], ["Normal", "Crackle", "Wheeze"]):
    idx = (y_val == label).nonzero(as_tuple=True)[0][0]
    sample = X_val[idx]
    visualize_feature_maps(model, sample, device, title_prefix=name)
'''