import os
import librosa
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences  # 패딩을 위한 함수

# 경로 설정
audio_dir = "data2/augmented_pitch/wheeze"
annotation_dir = "data2/annotations"
save_X = []
save_y = []

# MFCC 설정
n_mfcc = 13
target_sr = 4000
max_length = 253  # 원하는 최대 길이 (최대 시간 길이에 맞게 조정)

# 파일 목록
wavs = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])

for wav_file in tqdm(wavs):
    # pitch-shift된 파일: example_ps+2.wav → 원본 txt: example.txt
    base_name = wav_file.split("_ps")[0] + ".txt"
    txt_path = os.path.join(annotation_dir, base_name)
    wav_path = os.path.join(audio_dir, wav_file)

    if not os.path.exists(txt_path):
        print(f"❗ annotation 없음: {base_name}")
        continue

    # 오디오 로드
    y, sr = librosa.load(wav_path, sr=target_sr)

    # annotation 불러오기
    with open(txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            start, end, crackle, wheeze = map(float, line.strip().split())
            start_sample = int(start * sr)
            end_sample = int(end * sr)

            if end_sample > len(y):  # 범위 초과 방지
                continue

            segment = y[start_sample:end_sample]

            # 길이 짧은 건 건너뜀
            if len(segment) < int(0.3 * sr):
                continue

            # MFCC 변환
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)

            # 패딩 처리 (길이가 max_length에 맞춰서 자르거나 패딩)
            mfcc_padded = pad_sequences([mfcc.T], maxlen=max_length, padding='post', dtype='float32')

            save_X.append(mfcc_padded[0])
            save_y.append(2)  # Wheeze로 고정

# numpy 변환 및 저장
save_X = np.array(save_X)
save_y = np.array(save_y)

print(f"✅ 총 세그먼트 수: {len(save_X)}")
np.save("data2/X_wheeze_pitch_padded.npy", save_X)
np.save("data2/y_wheeze_pitch_padded.npy", save_y)
print("💾 저장 완료: X_wheeze_pitch_padded.npy, y_wheeze_pitch_padded.npy")