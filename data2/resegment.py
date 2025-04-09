import os
import librosa
import soundfile as sf
import pandas as pd

# 경로 설정
raw_audio_dir = "data/raw_audio"
annotation_dir = "data/annotations"
output_dir = "data2/segments_no_both"
os.makedirs(output_dir, exist_ok=True)

def segment_audio(audio_path, annotation_path, output_base):
    try:
        y, sr = librosa.load(audio_path, sr=4000)  # 4kHz로 통일
    except Exception as e:
        print(f"[ERROR] 오디오 로드 실패: {audio_path} → {e}")
        return

    try:
        df = pd.read_csv(annotation_path, sep="\t", header=None)
        df.columns = ["start", "end", "crackle", "wheeze"]
    except Exception as e:
        print(f"[ERROR] annotation 로드 실패: {annotation_path} → {e}")
        return

    saved = 0
    for i, row in df.iterrows():
        start, end, crackle, wheeze = row
        if crackle == 1 and wheeze == 1:
            continue  # Both인 경우 skip

        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = y[start_sample:end_sample]

        # 클래스 라벨 지정
        label = "normal"
        if crackle == 1:
            label = "crackle"
        elif wheeze == 1:
            label = "wheeze"

        segment_filename = f"{output_base}_{i}_{label}.wav"
        segment_path = os.path.join(output_dir, segment_filename)
        sf.write(segment_path, segment, sr)
        saved += 1

    print(f"[OK] {output_base}: {saved} segments 저장됨")

# 전체 오디오 파일 처리 루프
for filename in os.listdir(raw_audio_dir):
    if filename.endswith(".wav"):
        base_name = filename.replace(".wav", "")
        audio_path = os.path.join(raw_audio_dir, filename)
        annotation_path = os.path.join(annotation_dir, base_name + ".txt")

        if os.path.exists(annotation_path):
            segment_audio(audio_path, annotation_path, base_name)