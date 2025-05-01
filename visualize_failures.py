import torch
import matplotlib.pyplot as plt
import numpy as np

# 데이터 불러오기
X_val = torch.tensor(np.load("data3/split/X_val.npy")).float()
y_val = torch.tensor(np.load("data3/split/y_val.npy")).long()
thresholds = torch.tensor([0.44, 0.38, 0.42]) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_val = X_val.to(device)

from models.remastered_model import RemasteredCNNBiLSTM

model = RemasteredCNNBiLSTM(num_classes=3, dropout=0.35).to(device)
model.load_state_dict(torch.load("results/best_model.pth", map_location=device))
model.eval()


def predict_with_threshold(logits, thresholds):
    probs = torch.softmax(logits, dim=1)
    preds = []
    for row in probs:
        pred = (row >= thresholds).int()
        if pred.sum() == 0:
            pred[torch.argmax(row)] = 1
        preds.append(torch.argmax(pred).item())
    return torch.tensor(preds)

# 1. 모델 추론
model.eval()
with torch.no_grad():
    logits = model(X_val.to(device))
    preds = predict_with_threshold(logits, thresholds).cpu()
    y_true = y_val.cpu()

# 2. 실패한 인덱스 추출
wrong_idx = (preds != y_true).nonzero(as_tuple=True)[0]

# 3. 시각화 함수 정의
def plot_mfcc(index):
    mfcc = X_val[index].numpy()  # (13, 128)
    plt.figure(figsize=(6, 4))
    plt.imshow(mfcc, aspect='auto', origin='lower')
    plt.colorbar(label='MFCC Coefficient')
    plt.title(f"Predict failure: idx {index.item()}\nActual: {y_true[index].item()} / Prediction: {preds[index].item()}")
    plt.xlabel("Time Frame")
    plt.ylabel("MFCC Coefficients")
    plt.tight_layout()
    plt.show()

# 4. 상위 5개 예측 실패 샘플 시각화
for idx in wrong_idx[:5]:
    plot_mfcc(idx)
