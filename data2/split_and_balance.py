import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os

# 파일 로드
X = np.load("data2/mfcc/X_augmented_2.npy", allow_pickle=True)
y = np.load("data2/mfcc/y_augmented_2.npy")
metadata = pd.read_csv("data2/mfcc/metadata_2.csv")

# 환자 ID 추출
metadata["label"] = y  # label 컬럼 추가
metadata["patient_id"] = metadata["index"].apply(lambda x: f"{x // 100:03d}")

# 클래스별로 그룹핑
grouped = metadata.groupby("label")
target_per_class = 1600

selected_indices = []

for label, group in grouped:
    # 동일 환자에서 여러 샘플 뽑지 않도록 shuffle 후 그룹화
    patients = group["patient_id"].unique()
    np.random.shuffle(patients)

    collected = []
    for pid in patients:
        samples = group[group["patient_id"] == pid]
        collected.append(samples)
        if sum(len(c) for c in collected) >= target_per_class:
            break

    selected = pd.concat(collected).iloc[:target_per_class]
    selected_indices.extend(selected.index.tolist())

# 최종 데이터셋
balanced_meta = metadata.loc[selected_indices].reset_index(drop=True)
balanced_X = X[selected_indices]
balanced_y = y[selected_indices]

# 환자 기준 분할
patient_ids = balanced_meta["patient_id"].unique()
train_ids, temp_ids = train_test_split(patient_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

def get_split(ids):
    mask = balanced_meta["patient_id"].isin(ids)
    return balanced_X[mask], balanced_y[mask]

X_train, y_train = get_split(train_ids)
X_val, y_val = get_split(val_ids)
X_test, y_test = get_split(test_ids)

# 저장
np.save("data2/X_train.npy", X_train)
np.save("data2/y_train.npy", y_train)
np.save("data2/X_val.npy", X_val)
np.save("data2/y_val.npy", y_val)
np.save("data2/X_test.npy", X_test)
np.save("data2/y_test.npy", y_test)

print("✅ 클래스 균형 맞춤 + 환자 기준 분할 완료!")
print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

from collections import Counter

print("\n📊 클래스별 분포:")
print("Train:", Counter(y_train))
print("Val:", Counter(y_val))
print("Test:", Counter(y_test))