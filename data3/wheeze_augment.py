import os
import librosa
import soundfile as sf
from tqdm import tqdm

input_dir = "data3/segments/wheeze"
output_dir_time = "data3/segments_augmented/wheeze_time"
output_dir_pitch_up = "data3/segments_augmented/wheeze_pitch_up"
output_dir_pitch_down = "data3/segments_augmented/wheeze_pitch_down"
os.makedirs(output_dir_time, exist_ok=True)
os.makedirs(output_dir_pitch_up, exist_ok=True)
os.makedirs(output_dir_pitch_down, exist_ok=True)

sr_target = 4000
frame_threshold = 80  # 프레임 수가 이하면 time stretch 대상
hop_length = 128

for fname in tqdm(sorted(os.listdir(input_dir))):
    if not fname.endswith(".wav"):
        continue

    wav_path = os.path.join(input_dir, fname)
    signal, sr = librosa.load(wav_path, sr=sr_target)

    # MFCC 프레임 수 계산
    mfcc = librosa.feature.mfcc(y=signal, sr=sr_target, hop_length=hop_length, n_fft=512, n_mfcc=13)
    num_frames = mfcc.shape[1]

    # 1. Time Stretch (조건부 적용)
    if num_frames <= frame_threshold:
        stretched = librosa.effects.time_stretch(signal, rate=1.2)
        sf.write(os.path.join(output_dir_time, fname.replace(".wav", "_time.wav")), stretched, sr_target)

    # 2. Pitch Shift (전부 적용)
    pitch_up = librosa.effects.pitch_shift(signal, sr=sr_target, n_steps=2)
    pitch_down = librosa.effects.pitch_shift(signal, sr=sr_target, n_steps=-2)

    sf.write(os.path.join(output_dir_pitch_up, fname.replace(".wav", "_pu.wav")), pitch_up, sr_target)
    sf.write(os.path.join(output_dir_pitch_down, fname.replace(".wav", "_pd.wav")), pitch_down, sr_target)