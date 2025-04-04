import numpy as np
import pandas as pd
from sklearn.utils import resample

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")
meta = pd.read_csv("data/processed/metadata.csv")

# 클래스 구분
class_indices = {
    "00": np.where((y[:, 0] == 0) & (y[:, 1] == 0))[0],
    "10": np.where((y[:, 0] == 1) & (y[:, 1] == 0))[0],
    "01": np.where((y[:, 0] == 0) & (y[:, 1] == 1))[0],
    "11": np.where((y[:, 0] == 1) & (y[:, 1] == 1))[0],
}

target_size = 1500
resampled_indices = []

for key, indices in class_indices.items():
    if len(indices) > target_size:
        # 언더샘플링
        sampled = resample(indices, replace=False, n_samples=target_size, random_state=42)
    else:
        # 오버샘플링
        sampled = resample(indices, replace=True, n_samples=target_size, random_state=42)
    
    resampled_indices.extend(sampled)

# 최종 섞기
resampled_indices = np.array(resampled_indices)
np.random.seed(42)
np.random.shuffle(resampled_indices)

# Balanced dataset 만들기
X_bal = X[resampled_indices]
y_bal = y[resampled_indices]
meta_bal = meta.iloc[resampled_indices].reset_index(drop=True)

# 저장
np.save("data/processed/X_balanced.npy", X_bal)
np.save("data/processed/y_balanced.npy", y_bal)
meta_bal.to_csv("data/processed/metadata_balanced.csv", index=False)

print("균형 맞춘 데이터 저장 완료:", X_bal.shape, y_bal.shape)