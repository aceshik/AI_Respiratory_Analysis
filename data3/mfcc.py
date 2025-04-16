import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# 설정
segment_dir = "data3/segments"
output_dir = "data3/mfcc"
os.makedirs(output_dir, exist_ok=True)

sample_rate = 4000
n_mfcc = 13
n_fft = 512
hop_length = 128
max_len = 128

label_map = {"normal": 0, "crackle": 1, "wheeze": 2}

def pad_mfcc(mfcc, max_len=128):
    if mfcc.shape[1] < max_len:
        return np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        return mfcc[:, :max_len]

X = []
y = []
metadata = []

# 세그먼트 순회
for label_name in tqdm(["normal", "crackle", "wheeze"], desc="Processing labels"):
    label_id = label_map[label_name]
    folder_path = os.path.join(segment_dir, label_name)
    
    for fname in tqdm(sorted(os.listdir(folder_path)), desc=f"{label_name} files", leave=False):
        if not fname.endswith(".wav"):
            continue
        full_path = os.path.join(folder_path, fname)
        
        try:
            signal, sr = librosa.load(full_path, sr=sample_rate)
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc_padded = pad_mfcc(mfcc, max_len)

            patient_id = "_".join(fname.split("_")[:2])
            X.append(mfcc_padded)
            y.append(label_id)
            metadata.append({
                "filename": fname,
                "label": label_name,
                "label_id": label_id,
                "frame_length": mfcc.shape[1],
                "patient_id": patient_id,
            })

        except Exception as e:
            print(f"오류 발생: {fname} → {e}")

# 저장
np.save(os.path.join(output_dir, "X.npy"), np.array(X))
np.save(os.path.join(output_dir, "y.npy"), np.array(y))
pd.DataFrame(metadata).to_csv(os.path.join(output_dir, "metadata.csv"), index=False)