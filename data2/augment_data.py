import numpy as np
import os
import random
from collections import Counter
from sklearn.utils import resample
import pandas as pd

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
    import librosa
    # mfcc: (13, 253)
    # 역변환 후 stretch 후 다시 mfcc 추출 → 비현실적
    # 대신 간단하게 linear interpolation
    rate = np.random.uniform(*rate_range)
    new_len = int(mfcc.shape[1] * rate)
    stretched = np.zeros((mfcc.shape[0], new_len))
    for i in range(mfcc.shape[0]):
        stretched[i] = np.interp(np.linspace(0, 1, new_len),
                                 np.linspace(0, 1, mfcc.shape[1]),
                                 mfcc[i])
    # 다시 원래 길이로 resize
    if new_len < mfcc.shape[1]:
        pad_width = mfcc.shape[1] - new_len
        return np.pad(stretched, ((0,0), (0, pad_width)), mode='constant')
    else:
        return stretched[:, :mfcc.shape[1]]

def augment_mfcc(mfcc, n_aug=4):
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

# 메인
X = np.load("data2/mfcc/X.npy")
y = np.load("data2/mfcc/y.npy")

X_aug, y_aug = [], []
target_classes = [1, 2]  # Crackle, Wheeze

for i in range(len(X)):
    if y[i] in target_classes:
        aug_list = augment_mfcc(X[i], n_aug=2)  # 각 샘플당 2개씩 증강
        X_aug.extend(aug_list)
        y_aug.extend([y[i]] * len(aug_list))

X_final = np.concatenate([X, np.array(X_aug)], axis=0)
y_final = np.concatenate([y, np.array(y_aug)], axis=0)

print("증강 전:", X.shape, Counter(y))
print("증강 후:", X_final.shape, Counter(y_final))

np.save("data2/mfcc/X_augmented.npy", X_final)
np.save("data2/mfcc/y_augmented.npy", y_final)

# metadata.csv 저장
metadata = []
for i in range(len(X)):
    metadata.append({"index": i, "label": int(y[i]), "source": "original"})

for j in range(len(y_aug)):
    metadata.append({"index": len(X) + j, "label": int(y_aug[j]), "source": "augmented"})

df = pd.DataFrame(metadata)
df.to_csv("data2/mfcc/metadata.csv", index=False)
print("✅ metadata.csv 저장 완료:", df.shape)