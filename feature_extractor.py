import librosa
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
audio_path = os.path.join(BASE_DIR, "extracted_audio.wav")

print("🔊 Loading audio...")

# Load audio
y, sr = librosa.load(audio_path)

# -------- EXTRACT MFCC --------
print("📊 Extracting MFCC features...")
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# -------- EXTRACT ADDITIONAL FEATURES --------
# Zero crossing rate (how fast audio changes)
zcr = librosa.feature.zero_crossing_rate(y)

# Root mean square energy (loudness)
rms = librosa.feature.rms(y=y)

# Spectral centroid (brightness of sound)
spectral = librosa.feature.spectral_centroid(y=y, sr=sr)

print(f"✅ Features extracted!")
print(f"📊 MFCC shape: {mfcc.shape}")
print(f"📊 ZCR shape: {zcr.shape}")
print(f"📊 RMS shape: {rms.shape}")
print(f"📊 Spectral shape: {spectral.shape}")
print(f"🎵 Total audio frames: {mfcc.shape[1]}")
print(f"🎬 Video FPS: 30")
print(f"⏱️ Audio duration: {len(y)/sr:.2f} seconds")

# Save features
np.save("mfcc_features.npy", mfcc)
np.save("zcr_features.npy", zcr)
np.save("rms_features.npy", rms)

print("✅ Features saved as .npy files!")