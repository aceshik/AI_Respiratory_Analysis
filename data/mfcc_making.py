import os
import librosa
import numpy as np
import pandas as pd

input_wav_dir = "data/processed/resampled_wav"
input_txt_dir = "data/annotations"
output_dir = "data/processed"

os.makedirs(output_dir, exist_ok=True)

X_list = []
y_list = []
meta_list = []

target_sr = 4000
n_mfcc = 13
hop_length = 256
n_fft = 512

for filename in os.listdir(input_wav_dir):
    if not filename.endswith(".wav"):
        continue

    basename = filename.replace(".wav", "")
    wav_path = os.path.join(input_wav_dir, filename)
    txt_path = os.path.join(input_txt_dir, f"{basename}.txt")

    if not os.path.exists(txt_path):
        print(f"해당 annotation 없음: {txt_path}")
        continue

    y_audio, _ = librosa.load(wav_path, sr=target_sr)

    with open(txt_path, "r") as f:
        for i, line in enumerate(f):
            try:
                start, end, crackle, wheeze = line.strip().split()
                start_sample = int(float(start) * target_sr)
                end_sample = int(float(end) * target_sr)

                # 주기 단위로 자르기
                segment = y_audio[start_sample:end_sample]

                # 최소 길이 필터링 (너무 짧으면 스킵)
                if len(segment) < hop_length * 2:
                    continue

                # 정규화
                segment = segment / np.max(np.abs(segment))

                # MFCC 추출
                mfcc = librosa.feature.mfcc(
                    y=segment,
                    sr=target_sr,
                    n_mfcc=n_mfcc,
                    hop_length=hop_length,
                    n_fft=n_fft
                )

                X_list.append(mfcc)
                y_list.append([int(crackle), int(wheeze)])

                meta_list.append({
                    "filename": basename,
                    "cycle_index": i,
                    "start": start,
                    "end": end,
                    "crackle": crackle,
                    "wheeze": wheeze
                })

            except Exception as e:
                print(f"{basename} 주기 {i} 처리 실패: {e}")

# padding MFCC to same time_steps
max_len = max(m.shape[1] for m in X_list)
X_padded = np.array([
    np.pad(m, ((0, 0), (0, max_len - m.shape[1])), mode='constant')
    for m in X_list
])

y_array = np.array(y_list)
meta_df = pd.DataFrame(meta_list)

# 저장
np.save(os.path.join(output_dir, "X.npy"), X_padded)
np.save(os.path.join(output_dir, "y.npy"), y_array)
meta_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

print("저장 완료:", X_padded.shape, y_array.shape)