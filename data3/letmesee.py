import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 파일 경로
original_path = "data3/raw_audio/221_2b2_Pl_mc_LittC2SE.wav"
resampled_path = "data3/resampled_audio/221_2b2_Pl_mc_LittC2SE.wav"

# 로딩
y_orig, sr_orig = librosa.load(original_path, sr=None)
y_resamp, sr_resamp = librosa.load(resampled_path, sr=None)

# 시각화
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
librosa.display.waveshow(y_orig, sr=sr_orig)
plt.title(f"221_2b2_Pl_mc_LittC2SE.wav - Original ({sr_orig} Hz)")

plt.subplot(2, 1, 2)
librosa.display.waveshow(y_resamp, sr=sr_resamp)
plt.title(f"221_2b2_Pl_mc_LittC2SE.wav - Resampled ({sr_resamp} Hz)")

plt.tight_layout()
plt.show()

def plot_spectrogram(y, sr, title):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plot_spectrogram(y_orig, sr_orig, "221_2b2_Pl_mc_LittC2SE.wav - Spectrogram (Original)")

plt.subplot(2, 1, 2)
plot_spectrogram(y_resamp, sr_resamp, "221_2b2_Pl_mc_LittC2SE.wav - Spectrogram (Resampled)")

plt.tight_layout()
plt.show()

def plot_fft(y, sr, title):
    fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    plt.plot(freqs, np.abs(fft))
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plot_fft(y_orig, sr_orig, "221_2b2_Pl_mc_LittC2SE.wav - Frequency Spectrum (Original)")

plt.subplot(2, 1, 2)
plot_fft(y_resamp, sr_resamp, "221_2b2_Pl_mc_LittC2SE.wav - Frequency Spectrum (Resampled)")

plt.tight_layout()
plt.show()