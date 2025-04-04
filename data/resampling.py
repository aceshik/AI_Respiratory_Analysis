import os
import librosa
import soundfile as sf
import numpy as np

# 설정
input_dir = "data/raw_audio"
output_dir = "data/processed/resampled_wav"
target_sr = 4000

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.endswith(".wav"):
        continue
    
    filepath = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    try:
        # 원본 로드 (sr=None → 원래 샘플링 유지)
        y, orig_sr = librosa.load(filepath, sr=None)
        
        # 정규화: [-1, 1] 범위
        y = y / np.max(np.abs(y))
        
        # 리샘플링
        if orig_sr != target_sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)

        # 저장
        sf.write(output_path, y, target_sr)
        
    except Exception as e:
        print(f"오류 발생: {filename} → {e}")