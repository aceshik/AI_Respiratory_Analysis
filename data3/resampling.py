import os
import librosa
import soundfile as sf

input_dir = "data3/raw_audio"
output_dir = "data3/resampled_audio"
os.makedirs(output_dir, exist_ok=True)

target_sr = 4000  # 목표 샘플링 레이트

for fname in sorted(os.listdir(input_dir)):
    if not fname.endswith(".wav"):
        continue

    input_path = os.path.join(input_dir, fname)
    output_path = os.path.join(output_dir, fname)

    try:
        # 원래 SR로 로딩
        y, sr = librosa.load(input_path, sr=None)

        # 리샘플링
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        # 저장
        sf.write(output_path, y_resampled, target_sr)
        print(f"{fname}: {sr} Hz → {target_sr} Hz 리샘플링 완료")
    except Exception as e:
        print(f"{fname}: 오류 발생 → {e}")