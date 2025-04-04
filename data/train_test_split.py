import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 로드
meta = pd.read_csv("data/processed/metadata_balanced.csv")
X = np.load("data/processed/X_balanced.npy")
y = np.load("data/processed/y_balanced.npy")

# 환자 ID 추출
meta["patient_id"] = meta["filename"].apply(lambda x: x.split("_")[0])

# 환자 목록 분리
unique_patients = meta["patient_id"].unique()
train_ids, testval_ids = train_test_split(unique_patients, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(testval_ids, test_size=0.5, random_state=42)

# 인덱스 선택
train_idx = meta[meta["patient_id"].isin(train_ids)].index
val_idx = meta[meta["patient_id"].isin(val_ids)].index
test_idx = meta[meta["patient_id"].isin(test_ids)].index

# 데이터 분리
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# 저장
np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/X_val.npy", X_val)
np.save("data/processed/y_val.npy", y_val)
np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_test.npy", y_test)

print("스플릿 완료:")
print(f"Train: {len(train_idx)} samples")
print(f"Val:   {len(val_idx)} samples")
print(f"Test:  {len(test_idx)} samples")