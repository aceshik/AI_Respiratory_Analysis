import numpy as np
from collections import Counter
import random
import os

# 데이터 로드
X = np.load("data2/mfcc/X.npy")
y = np.load("data2/mfcc/y.npy")

# 클래스별 인덱스 추출
indices = {i: np.where(y == i)[0].tolist() for i in np.unique(y)}
print("원본 클래스 분포:", {k: len(v) for k, v in indices.items()})

# 원하는 수로 샘플 개수 통일 (가장 작은 886 기준)
target_count = 100  # 원하는 균형 수

balanced_indices = []

for class_id, idx_list in indices.items():
    if len(idx_list) > target_count:
        # undersampling
        sampled = random.sample(idx_list, target_count)
    else:
        # oversampling
        sampled = np.random.choice(idx_list, target_count, replace=True).tolist()
    balanced_indices.extend(sampled)

# 셔플
random.shuffle(balanced_indices)

# 결과 생성 및 저장
X_balanced = X[balanced_indices]
y_balanced = y[balanced_indices]

print("균형 클래스 분포:", Counter(y_balanced))

np.save("data2/mfcc/X_balanced.npy", X_balanced)
np.save("data2/mfcc/y_balanced.npy", y_balanced)