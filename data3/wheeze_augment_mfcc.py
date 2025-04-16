import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# 증강된 오디오 폴더
aug_dirs = [
    "data3/segments_augmented/wheeze_time",
    "data3/segments_augmented/wheeze_pitch_up",
    "data3/segments_augmented/wheeze_pitch_down"
]

# Split 정보 로드 (환자 기준)
meta_split = pd.read_csv("data3/split/metadata_split.csv")
split_dict = dict(zip(meta_split["patient_id"], meta_split["split"]))

# MFCC 설정
sr = 4000
n_mfcc = 13
n_fft = 512
hop_length = 128
max_len = 128

def pad_mfcc(mfcc, max_len):
    if mfcc.shape[1] < max_len:
        return np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        return mfcc[:, :max_len]

# 저장할 리스트
X, y, metadata = [], [], []

for aug_dir in aug_dirs:
    for fname in tqdm(sorted(os.listdir(aug_dir))):
        if not fname.endswith(".wav"):
            continue
        file_path = os.path.join(aug_dir, fname)

        # 파일명에서 환자 ID 추출 (예: 221_2b2_004_wheeze_pu.wav → 221_2b2)
        base = fname.split("_")
        patient_id = "_".join(base[:2])
        label_name = "wheeze"
        label_id = 2
        split = split_dict.get(patient_id, "unknown")

        try:
            signal, _ = librosa.load(file_path, sr=sr)
            signal = signal / np.max(np.abs(signal))
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc = pad_mfcc(mfcc, max_len)
            X.append(mfcc)
            y.append(label_id)
            metadata.append({
                "filename": fname,
                "label": label_name,
                "label_id": label_id,
                "patient_id": patient_id,
                "split": split
            })
        except Exception as e:
            print(f"오류 발생: {fname} → {e}")

# 저장
output_dir = "data3/mfcc_augmented"
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "X_wheeze_aug.npy"), np.array(X))
np.save(os.path.join(output_dir, "y_wheeze_aug.npy"), np.array(y))
pd.DataFrame(metadata).to_csv(os.path.join(output_dir, "metadata_wheeze_aug.csv"), index=False)

print("✅ Wheeze 증강 데이터 MFCC 추출 완료")