import os
import numpy as np
import librosa
import soundfile as sf
import random

# 설정
DATA_DIR = "data/raw_audio"
SR = 16000
FIXED_LEN = 253
AUGMENT_PER_LABEL = {
    0: 0,  # Normal
    1: 2,  # Crackle
    2: 2,  # Wheeze
    3: 3,  # Crackle + Wheeze
}

# 증강 함수
def add_white_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise

def time_stretch(audio, rate=1.1):
    return librosa.effects.time_stretch(audio, rate)

def pitch_shift(audio, sr, steps=2):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)

def extract_mfcc(audio, sr, n_mfcc=13, fixed_len=FIXED_LEN):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < fixed_len:
        pad = fixed_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
    else:
        mfcc = mfcc[:, :fixed_len]
    return mfcc

# segment + 증강 + MFCC 추출
def process_one_file(wav_path, txt_path):
    audio, original_sr = librosa.load(wav_path, sr=None)
    audio = librosa.resample(audio, orig_sr=original_sr, target_sr=SR)

    segments = []
    labels = []

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            start, end = float(parts[0]), float(parts[1])
            crackle, wheeze = int(parts[2]), int(parts[3])

            if crackle == 0 and wheeze == 0:
                label = 0  # Normal
            elif crackle == 1 and wheeze == 0:
                label = 1  # Crackle
            elif crackle == 0 and wheeze == 1:
                label = 2  # Wheeze
            else:
                label = 3  # Crackle + Wheeze

            start_idx = int(start * SR)
            end_idx = int(end * SR)
            segment = audio[start_idx:end_idx]

            if len(segment) < 100:  # 너무 짧은 구간은 skip
                continue

            # 원본
            mfcc = extract_mfcc(segment, SR)
            segments.append(mfcc)
            labels.append(label)

            # 증강
            for _ in range(AUGMENT_PER_LABEL.get(label, 0)):
                aug_type = random.choice(['noise', 'stretch', 'pitch'])

                try:
                    if aug_type == 'noise':
                        aug_seg = add_white_noise(segment)
                    elif aug_type == 'stretch':
                        aug_seg = time_stretch(segment, rate=np.random.uniform(0.9, 1.1))
                    elif aug_type == 'pitch':
                        aug_seg = pitch_shift(segment, SR, steps=random.choice([-2, -1, 1, 2]))
                    else:
                        continue
                    mfcc_aug = extract_mfcc(aug_seg, SR)
                    segments.append(mfcc_aug)
                    labels.append(label)
                except:
                    continue  # 증강 실패 시 skip

    return segments, labels

SAVE_DIR = "data/augmentation/augmented"
os.makedirs(SAVE_DIR, exist_ok=True)

# 전체 처리
def run_full_pipeline():
    all_segments = []
    all_labels = []

    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".wav"):
            continue
        wav_path = os.path.join(DATA_DIR, fname)
        txt_filename = os.path.splitext(os.path.basename(wav_path))[0] + ".txt"
        txt_path = os.path.join("data/annotations", txt_filename)

        if not os.path.exists(txt_path):
            print(f"⚠️ annotation 파일 없음: {txt_path}")
            continue

        print(f"▶ 처리 중: {fname}")
        segs, labs = process_one_file(wav_path, txt_path)
        all_segments.extend(segs)
        all_labels.extend(labs)

    X = np.array(all_segments)
    y = np.array(all_labels)

    print(f"\n✅ 총 샘플 수: {len(y)} (shape: {X.shape})")

    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(os.path.join(SAVE_DIR, "X_all_augmented_advanced.npy"), X)
    np.save(os.path.join(SAVE_DIR, "y_all_augmented_advanced.npy"), y)
    print("💾 저장 완료: X_all_augmented_advanced.npy / y_all_augmented_advanced.npy")

# 실행
if __name__ == "__main__":
    run_full_pipeline()