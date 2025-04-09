import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

segment_dir = "data2/segments_no_both"
output_dir = "data2/mfcc"
os.makedirs(output_dir, exist_ok=True)

label_map = {"normal": 0, "crackle": 1, "wheeze": 2}

def pad_mfcc(mfcc, max_len=253):
    if mfcc.shape[1] < max_len:
        return np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        return mfcc[:, :max_len]

X, y, metadata = [], [], []

for fname in tqdm(sorted(os.listdir(segment_dir))):
    if not fname.endswith(".wav"):
        continue

    label_name = fname.split("_")[-1].replace(".wav", "")
    label = label_map.get(label_name)
    if label is None:
        continue

    wav_path = os.path.join(segment_dir, fname)
    try:
        signal, sr = librosa.load(wav_path, sr=4000)
        signal = signal / np.max(np.abs(signal))  # 정규화
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        mfcc_padded = pad_mfcc(mfcc)
        X.append(mfcc_padded)
        y.append(label)
        metadata.append({
            "filename": fname,
            "label": label_name,
            "label_id": label,
            "length": len(signal),
            "sr": sr
        })
    except Exception as e:
        print(f"오류 발생: {fname} → {e}")

np.save(os.path.join(output_dir, "X.npy"), np.array(X))
np.save(os.path.join(output_dir, "y.npy"), np.array(y))
pd.DataFrame(metadata).to_csv(os.path.join(output_dir, "metadata.csv"), index=False)