import os
import librosa
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

segment_dir = "data3/segments/crackle"
output_dir = "data3/mfcc_augmented"
os.makedirs(output_dir, exist_ok=True)

sample_rate = 4000
n_mfcc = 13
max_len = 128
label_id = 1  # Crackle

def pad_or_trim(mfcc, max_len=128):
    if mfcc.shape[1] < max_len:
        return np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        return mfcc[:, :max_len]

def add_noise(mfcc, noise_std=0.05):
    noise = np.random.randn(*mfcc.shape) * noise_std
    return mfcc + noise

def load_and_extract(path):
    y, sr = librosa.load(path, sr=sample_rate)
    y = y / np.max(np.abs(y))  # 정규화
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

# 모든 crackle segment 로드
files = sorted([f for f in os.listdir(segment_dir) if f.endswith('.wav')])
random.shuffle(files)

X_aug, y_aug, metadata = [], [], []
target_aug_count = 1000  # 조합 후 총 샘플 수
combo_size = 2

print(f"총 crackle 세그먼트 수: {len(files)} → 증강 목표: {target_aug_count}")

for i in tqdm(range(target_aug_count)):
    selected = random.sample(files, combo_size)
    combined_mfcc = []

    for fname in selected:
        path = os.path.join(segment_dir, fname)
        mfcc = load_and_extract(path)
        combined_mfcc.append(mfcc)

    merged = np.concatenate(combined_mfcc, axis=1)
    merged = pad_or_trim(merged, max_len=max_len)
    merged = add_noise(merged)

    X_aug.append(merged)
    y_aug.append(label_id)

    metadata.append({
        "filename": f"crackle_aug_{i:04d}.wav",
        "label": "crackle",
        "label_id": label_id,
        "source": " + ".join(selected),
        "patient_id": selected[0].split("_")[0] + "_" + selected[0].split("_")[1]
    })

# 저장
np.save(os.path.join(output_dir, "X_crackle_aug.npy"), np.array(X_aug))
np.save(os.path.join(output_dir, "y_crackle_aug.npy"), np.array(y_aug))

# split 정보 병합
split_info = pd.read_csv("data3/split/metadata_split.csv")[["patient_id", "split"]]
split_map = dict(zip(split_info["patient_id"], split_info["split"]))
for m in metadata:
    m["split"] = split_map.get(m["patient_id"], "unknown")
    del m["source"]
pd.DataFrame(metadata).to_csv(os.path.join(output_dir, "metadata_crackle_aug.csv"), index=False)

print("✅ Crackle 증강 데이터 저장 완료.")
