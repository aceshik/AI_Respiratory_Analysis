import os
import librosa
import soundfile as sf
from tqdm import tqdm

# 경로 설정
annotation_dir = "data2/annotations"
audio_dir = "data2/raw_audio"
output_dir = "data2/augmented_pitch/wheeze"
os.makedirs(output_dir, exist_ok=True)

# Wheeze가 있는 오디오 파일 자동 탐색
wheeze_audio_files = []

for ann_file in os.listdir(annotation_dir):
    if not ann_file.endswith(".txt"):
        continue
    ann_path = os.path.join(annotation_dir, ann_file)
    with open(ann_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[3] == "1":
                wheeze_audio_files.append(ann_file.replace(".txt", ".wav"))
                break  # 하나라도 있으면 해당 파일은 추가

print(f"💡 Wheeze 파일 수: {len(wheeze_audio_files)}개")

# pitch shifting 수행
target_sr = 4000
semitone_shifts = [-2, 2]

for filename in tqdm(wheeze_audio_files, desc="Pitch shifting wheeze"):
    filepath = os.path.join(audio_dir, filename)
    
    try:
        y, sr = librosa.load(filepath, sr=target_sr)  # 리샘플링
        for shift in semitone_shifts:
            y_shifted = librosa.effects.pitch_shift(y, sr=target_sr, n_steps=shift)
            new_filename = filename.replace(".wav", f"_ps{shift:+}.wav")
            save_path = os.path.join(output_dir, new_filename)
            sf.write(save_path, y_shifted, target_sr)
    except Exception as e:
        print(f"⚠️ {filename} 처리 중 오류 발생: {e}")