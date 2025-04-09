import numpy as np
import random
from collections import Counter
from sklearn.utils import resample
import pandas as pd

# 증강 함수들
def add_noise(mfcc, noise_level=0.01):
    noise = np.random.randn(*mfcc.shape) * noise_level
    return mfcc + noise

def time_mask(mfcc, max_mask_size=20):
    mfcc = mfcc.copy()
    t = mfcc.shape[1]
    mask_size = random.randint(5, max_mask_size)
    mask_start = random.randint(0, t - mask_size)
    mfcc[:, mask_start:mask_start + mask_size] = 0
    return mfcc

def time_stretch(mfcc, rate_range=(0.8, 1.2)):
    rate = np.random.uniform(*rate_range)
    new_len = int(mfcc.shape[1] * rate)
    stretched = np.zeros((mfcc.shape[0], new_len))
    for i in range(mfcc.shape[0]):
        stretched[i] = np.interp(np.linspace(0, 1, new_len),
                                 np.linspace(0, 1, mfcc.shape[1]),
                                 mfcc[i])
    if new_len < mfcc.shape[1]:
        pad_width = mfcc.shape[1] - new_len
        return np.pad(stretched, ((0, 0), (0, pad_width)), mode='constant')
    else:
        return stretched[:, :mfcc.shape[1]]

def augment_mfcc(mfcc, n_aug=2):
    augmented = []
    for _ in range(n_aug):
        aug = mfcc.copy()
        if random.random() < 0.5:
            aug = add_noise(aug)
        if random.random() < 0.5:
            aug = time_mask(aug)
        if random.random() < 0.5:
            aug = time_stretch(aug)
        augmented.append(aug)
    return augmented

# 기존 증강 데이터 불러오기
X = np.load("data2/mfcc/X_augmented.npy")
y = np.load("data2/mfcc/y_augmented.npy")

print("💡 증강 전:", X.shape, Counter(y))

X_crackle = X[y == 1]
y_crackle = y[y == 1]

X_aug, y_aug = [], []
for i in range(len(X_crackle)):
    augmented = augment_mfcc(X_crackle[i], n_aug=1)  # 1개씩만 추가
    X_aug.extend(augmented)
    y_aug.extend([1] * len(augmented))

# 합치기
X_new = np.concatenate([X, np.array(X_aug)], axis=0)
y_new = np.concatenate([y, np.array(y_aug)], axis=0)

np.save("data2/mfcc/X_augmented_2.npy", X_new)
np.save("data2/mfcc/y_augmented_2.npy", y_new)

print("✅ Crackle 추가 증강 완료:", X_new.shape, Counter(y_new))

# metadata.csv 저장
metadata = []
n_orig = len(X)
n_aug = len(y_aug)

for i in range(n_orig):
    metadata.append({"index": i, "label": int(y[i]), "source": "original"})

for j in range(n_aug):
    metadata.append({"index": n_orig + j, "label": int(y_aug[j]), "source": "augmented"})

df = pd.DataFrame(metadata)
df.to_csv("data2/mfcc/metadata_2.csv", index=False)
print("✅ metadata_2.csv 저장 완료:", df.shape)