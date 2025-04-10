import numpy as np
import torch
from sklearn.metrics import classification_report
from models.enhanced_model import EnhancedCNNEnsemble

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Threshold 및 클래스 이름
thresholds = [0.5, 0.45, 0.4]
class_names = ["Normal", "Crackle", "Wheeze"]

# 데이터 로드
X_val = np.load("data2/X_val.npy")        # (B, 13, 253)
y_val = np.load("data2/y_val.npy")        # (B,)

# 정규화를 위한 mean/std 로드
X_mean, X_std = np.load("data2/X_mean_std.npy", allow_pickle=True)

# 정규화 적용 (train과 동일하게)
X_val = (X_val - X_mean) / X_std

# Tensor 변환 및 device 이동
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

# 모델 불러오기
model = EnhancedCNNEnsemble().to(device)
model.load_state_dict(torch.load("results/best_model.pth", map_location=device))
model.eval()

# 추론
with torch.no_grad():
    outputs = model(X_val_tensor)
    probs = torch.softmax(outputs, dim=1)

# Threshold 적용
preds = []
for p in probs:
    passed = (p >= torch.tensor(thresholds, device=device)).nonzero(as_tuple=True)[0]
    if len(passed) == 0:
        pred = torch.argmax(p)
    else:
        pred = passed[torch.argmax(p[passed])]
    preds.append(pred.item())

# 평가
print("📌 Inference 결과 (Threshold 적용)")
print(classification_report(y_val_tensor.cpu(), preds, target_names=class_names, zero_division=0))