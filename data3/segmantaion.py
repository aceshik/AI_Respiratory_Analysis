import os
import librosa
import soundfile as sf
import pandas as pd
metadata_records = []

audio_dir = "data3/resampled_audio"
anno_dir = "data3/annotation"
segment_base = "data3/segments"

os.makedirs(segment_base, exist_ok=True)
for cls in ["normal", "crackle", "wheeze", "both"]:
    os.makedirs(os.path.join(segment_base, cls), exist_ok=True)

sr = 4000
segment_counter = 0

for anno_file in sorted(os.listdir(anno_dir)):
    if not anno_file.endswith(".txt"):
        continue

    wav_name = anno_file.replace(".txt", ".wav")
    wav_path = os.path.join(audio_dir, wav_name)
    anno_path = os.path.join(anno_dir, anno_file)

    if not os.path.exists(wav_path):
        continue

    y, _ = librosa.load(wav_path, sr=sr)
    patient_id = wav_name.split("_")[0] + "_" + wav_name.split("_")[1]
    patient_segment_counter = 0

    with open(anno_path, "r") as f:
        for line in f:
            start, end, crackle, wheeze = line.strip().split()
            crackle, wheeze = int(crackle), int(wheeze)

            if crackle == 1 and wheeze == 1:
                label = "both"
            elif crackle == 1:
                label = "crackle"
            elif wheeze == 1:
                label = "wheeze"
            else:
                label = "normal"

            start_idx = int(float(start) * sr)
            end_idx = int(float(end) * sr)
            segment = y[start_idx:end_idx]

            out_path = os.path.join(segment_base, label, f"{patient_id}_{patient_segment_counter:03d}_{label}.wav")
            sf.write(out_path, segment, sr)
            patient_segment_counter += 1
            
            metadata_records.append({
                "filename": f"{patient_id}_{patient_segment_counter:03d}_{label}.wav",
                "label": label,
                "label_id": {"normal": 0, "crackle": 1, "wheeze": 2, "both": 3}[label],
                "patient_id": patient_id,
                "start": float(start),
                "end": float(end),
                "sample_length": len(segment)
            })

metadata_df = pd.DataFrame(metadata_records)
metadata_df.to_csv(os.path.join(segment_base, "metadata.csv"), index=False)