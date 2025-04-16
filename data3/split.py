import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 경로 설정
mfcc_dir = "data3/mfcc"
output_dir = "data3/split"
os.makedirs(output_dir, exist_ok=True)

# 데이터 로드
X = np.load(os.path.join(mfcc_dir, "X.npy"))
y = np.load(os.path.join(mfcc_dir, "y.npy"))
meta = pd.read_csv(os.path.join(mfcc_dir, "metadata.csv"))

# 환자 ID 추출
meta["patient_id"] = meta["filename"].apply(lambda x: "_".join(x.split("_")[:2]))
unique_patients = meta["patient_id"].unique()

# Train 70%, Temp 30%
train_patients, temp_patients = train_test_split(unique_patients, test_size=0.3, random_state=42)
# Val/Test 각각 15%
val_patients, test_patients = train_test_split(temp_patients, test_size=0.5, random_state=42)

# Split 라벨 할당
meta["split"] = meta["patient_id"].apply(
    lambda pid: "train" if pid in train_patients else "val" if pid in val_patients else "test"
)

# 인덱스 추출
train_idx = meta[meta["split"] == "train"].index
val_idx = meta[meta["split"] == "val"].index
test_idx = meta[meta["split"] == "test"].index

# 데이터 분할
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val     = X[val_idx], y[val_idx]
X_test, y_test   = X[test_idx], y[test_idx]

# 저장
np.save(os.path.join(output_dir, "X_train.npy"), X_train)
np.save(os.path.join(output_dir, "y_train.npy"), y_train)
np.save(os.path.join(output_dir, "X_val.npy"), X_val)
np.save(os.path.join(output_dir, "y_val.npy"), y_val)
np.save(os.path.join(output_dir, "X_test.npy"), X_test)
np.save(os.path.join(output_dir, "y_test.npy"), y_test)

# 메타데이터도 저장
meta.to_csv(os.path.join(output_dir, "metadata_split.csv"), index=False)

print("✅ 데이터가 환자 기준으로 분할되어 저장되었습니다.")

print("\n--- 데이터 분할 요약 ---")
print("Train 개수:", len(train_idx))
print(meta.loc[train_idx, "label"].value_counts(), "\n")

print("Validation 개수:", len(val_idx))
print(meta.loc[val_idx, "label"].value_counts(), "\n")

print("Test 개수:", len(test_idx))
print(meta.loc[test_idx, "label"].value_counts())