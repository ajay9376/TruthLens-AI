"""
TruthLens AI — Voice Clone Detector v2.0
=========================================
Detects if a voice is AI generated or cloned using
audio feature analysis with LibROSA.

Key signals:
  ① Spectral Flatness  — AI voices are too flat (weight: 60%)  ← DOMINANT
  ② Pitch Consistency  — AI voices are too consistent (weight: 15%)
  ③ MFCC Variance      — AI voices lack natural variation (weight: 15%)
  ④ ZCR Variation      — AI voices have unnatural noise (weight: 5%)
  ⑤ Bandwidth Variation — AI voices lack bandwidth variety (weight: 5%)
"""

import librosa
import numpy as np
import subprocess
import tempfile
import os
import sys
import io

# Fix emoji output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FFMPEG_PATH = r"C:\Users\gujju\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin"


def add_ffmpeg():
    if os.path.exists(FFMPEG_PATH):
        os.environ["PATH"] += os.pathsep + FFMPEG_PATH
    os.environ["PATH"] += ":/usr/bin:/usr/local/bin"


def extract_audio(video_path: str) -> str:
    tmp = tempfile.mktemp(suffix="_voice.wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "22050", "-ac", "1", tmp
    ]
    subprocess.run(cmd, capture_output=True)
    return tmp if os.path.exists(tmp) else None


def analyze_voice(audio_path: str) -> dict:
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # ① Spectral Flatness
    spec_flatness = librosa.feature.spectral_flatness(y=y)[0]
    avg_flatness  = float(np.mean(spec_flatness))
    std_flatness  = float(np.std(spec_flatness))

    # ② Pitch Analysis
    f0 = librosa.yin(y, fmin=50, fmax=400)
    f0_voiced = f0[f0 > 0]
    if len(f0_voiced) > 0:
        pitch_mean = float(np.mean(f0_voiced))
        pitch_std  = float(np.std(f0_voiced))
        pitch_cov  = pitch_std / (pitch_mean + 1e-6)
    else:
        pitch_mean = 0
        pitch_std  = 0
        pitch_cov  = 0

    # ③ MFCC Variance
    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_var = float(np.mean(np.var(mfcc, axis=1)))

    # ④ Zero Crossing Rate
    zcr     = librosa.feature.zero_crossing_rate(y)[0]
    zcr_std = float(np.std(zcr))

    # ⑤ Spectral Bandwidth
    bandwidth     = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    bandwidth_std = float(np.std(bandwidth))

    return {
        'avg_flatness':   avg_flatness,
        'std_flatness':   std_flatness,
        'pitch_mean':     pitch_mean,
        'pitch_std':      pitch_std,
        'pitch_cov':      pitch_cov,
        'mfcc_var':       mfcc_var,
        'zcr_std':        zcr_std,
        'bandwidth_std':  bandwidth_std,
    }


def score_voice(metrics: dict) -> float:
    score = 0

    print(f"\n📊 Voice Analysis Metrics:")
    print(f"   Spectral Flatness : {metrics['avg_flatness']:.4f} (real<0.01, AI>0.02)")
    print(f"   Flatness Std      : {metrics['std_flatness']:.4f}")
    print(f"   Pitch CoV         : {metrics['pitch_cov']:.3f} (real>0.15, AI<0.10)")
    print(f"   MFCC Variance     : {metrics['mfcc_var']:.2f} (real>50, AI<30)")
    print(f"   ZCR Std           : {metrics['zcr_std']:.4f}")
    print(f"   Bandwidth Std     : {metrics['bandwidth_std']:.2f}")

    # ① Spectral Flatness (max 60 pts) ← DOMINANT signal
    if metrics['avg_flatness'] < 0.008:
        pts = 60
    elif metrics['avg_flatness'] < 0.015:
        pts = 40
    elif metrics['avg_flatness'] < 0.025:
        pts = 15
    else:
        pts = 0   # AI voice = 0 pts
    score += pts
    print(f"\n   ① Spectral Flatness {metrics['avg_flatness']:.4f} → +{pts}/60 pts")

    # ② Pitch Variation (max 15 pts)
    if metrics['pitch_cov'] >= 0.20:
        pts = 15
    elif metrics['pitch_cov'] >= 0.12:
        pts = 8
    else:
        pts = 0
    score += pts
    print(f"   ② Pitch CoV {metrics['pitch_cov']:.3f} → +{pts}/15 pts")

    # ③ MFCC Variance (max 15 pts)
    if metrics['mfcc_var'] >= 80:
        pts = 15
    elif metrics['mfcc_var'] >= 50:
        pts = 8
    else:
        pts = 0
    score += pts
    print(f"   ③ MFCC Variance {metrics['mfcc_var']:.2f} → +{pts}/15 pts")

    # ④ ZCR Variation (max 5 pts)
    if metrics['zcr_std'] >= 0.05:
        pts = 5
    elif metrics['zcr_std'] >= 0.03:
        pts = 2
    else:
        pts = 0
    score += pts
    print(f"   ④ ZCR Std {metrics['zcr_std']:.4f} → +{pts}/5 pts")

    # ⑤ Bandwidth Variation (max 5 pts)
    if metrics['bandwidth_std'] >= 300:
        pts = 5
    elif metrics['bandwidth_std'] >= 150:
        pts = 2
    else:
        pts = 0
    score += pts
    print(f"   ⑤ Bandwidth Std {metrics['bandwidth_std']:.2f} → +{pts}/5 pts")

    total = min(int(score), 100)
    print(f"\n   TOTAL: {total}/100")
    return float(total)


def analyze_voice_clone(video_or_audio_path: str):
    print("\n" + "="*50)
    print("🎙️  TruthLens AI — Voice Clone Detector v2.0")
    print("="*50)

    add_ffmpeg()

    path    = video_or_audio_path
    tmp_wav = None

    if video_or_audio_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print("🎬 Extracting audio from video...")
        tmp_wav = extract_audio(video_or_audio_path)
        if tmp_wav is None:
            print("❌ Audio extraction failed!")
            return 50.0, "UNKNOWN"
        path = tmp_wav
        print("✅ Audio extracted!")

    print("🎵 Analyzing voice features...")

    try:
        metrics = analyze_voice(path)
        score   = score_voice(metrics)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        if tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)
        return 50.0, "UNKNOWN"

    if tmp_wav and os.path.exists(tmp_wav):
        os.remove(tmp_wav)

    print(f"\n🎯 Voice Clone Score: {score:.1f}/100")

    if score >= 60:
        print("✅ Voice: REAL — Natural human voice!")
        verdict = "REAL"
    elif score >= 42:
        print("⚠️ Voice: SUSPICIOUS — Unusual voice patterns!")
        verdict = "SUSPICIOUS"
    else:
        print("❌ Voice: CLONED — AI generated voice detected!")
        verdict = "FAKE"

    print("="*50)
    return score, verdict


# ─── CLI ───
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "test_video.mp4")
    score, verdict = analyze_voice_clone(path)
    print(f"\n🏁 Final: {verdict} ({score:.1f}/100)")