import numpy as np
import os

# 경로 설정
base_path = "data/processed"

# 세트별 데이터 로드
X_train = np.load(os.path.join(base_path, "X_train.npy"))
X_val = np.load(os.path.join(base_path, "X_val.npy"))
X_test = np.load(os.path.join(base_path, "X_test.npy"))

# 전체 데이터 기준 평균과 표준편차 계산
X_all = np.concatenate([X_train, X_val, X_test], axis=0)
mean = X_all.mean()
std = X_all.std()

print(f"전체 평균: {mean:.4f}, 전체 표준편차: {std:.4f}")

# 정규화 적용
X_train_norm = (X_train - mean) / std
X_val_norm = (X_val - mean) / std
X_test_norm = (X_test - mean) / std

# 정규화된 데이터 저장
np.save(os.path.join(base_path, "X_train_norm.npy"), X_train_norm)
np.save(os.path.join(base_path, "X_val_norm.npy"), X_val_norm)
np.save(os.path.join(base_path, "X_test_norm.npy"), X_test_norm)

print("정규화 및 저장 완료")