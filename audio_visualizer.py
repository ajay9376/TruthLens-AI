import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
audio_path = os.path.join(BASE_DIR, "extracted_audio.wav")

print("🔊 Loading audio...")

# Load audio
y, sr = librosa.load(audio_path)

# -------- WAVEFORM --------
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title("1. Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

# -------- SPECTROGRAM --------
D = librosa.stft(y)
D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.subplot(3, 1, 2)
librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title("2. Spectrogram")

# -------- MFCC --------
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

plt.subplot(3, 1, 3)
librosa.display.specshow(mfcc, sr=sr, x_axis='time')
plt.colorbar()
plt.title("3. MFCC (Speech Features)")

plt.tight_layout()
plt.savefig("audio_analysis.png")
plt.show()

print("✅ Waveform, Spectrogram and MFCC saved!")
print("📁 Saved as: audio_analysis.png")