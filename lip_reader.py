"""
TruthLens AI — Lip Reader / Visual Speech Analyzer  v2.0
=========================================================
Analyzes lip movement patterns to distinguish real speech from
AI-generated video.

Key Insight (from ground-truth calibration)
--------------------------------------------
AI generators OVER-animate lips — they are TOO well-synced, too rhythmic,
too exaggerated. Real selfie-camera speech is more natural and moderate.

  AI video:   MAR CoV=88%, voiced/silence ratio=1.59, band_power=0.47
  Real video: MAR CoV=50%, voiced/silence ratio=1.08, band_power=0.40

Signals (0-100, higher = more REAL)
--------------------------------------
  ① Delta Correlation      35 pts   DELTA-MAR vs DELTA-audio energy
  ② Voiced/Silence Ratio   25 pts   AI over-animates (ratio > 1.4)
  ③ Lip Rhythm Band Power  25 pts   AI is over-rhythmic (>0.46)
  ④ Velocity Asymmetry     15 pts   AI has unnatural open/close speed
"""

import sys
import io
import os
import cv2
import numpy as np
import mediapipe as mp
import subprocess
import tempfile
import librosa
from scipy.signal import welch

# Fix emoji output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FFMPEG_PATH = r"C:\Users\gujju\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin"

# MediaPipe (lazy init)
mp_face_mesh = mp.solutions.face_mesh
_face_mesh   = None

def _get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
    return _face_mesh

INNER_TOP = 13; INNER_BOTTOM = 14; OUTER_LEFT = 61; OUTER_RIGHT = 291


# ─────────────────────────────────────────────────
#  Audio
# ─────────────────────────────────────────────────

def extract_audio(video_path):
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + FFMPEG_PATH
    temp_wav = tempfile.mktemp(suffix="_tl.wav")
    cmd = ["ffmpeg", "-y", "-i", video_path,
           "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", temp_wav]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return temp_wav if (r.returncode == 0 and os.path.exists(temp_wav)) else None


def get_audio_energy_per_frame(wav_path, fps, total_frames):
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    hop   = max(1, int(sr / fps))
    rms   = librosa.feature.rms(y=y, hop_length=hop, frame_length=hop * 4)[0]
    if len(rms) != total_frames:
        rms = np.interp(np.linspace(0, len(rms)-1, total_frames),
                        np.arange(len(rms)), rms)
    return rms.astype(np.float32)


# ─────────────────────────────────────────────────
#  Lip tracking
# ─────────────────────────────────────────────────

def compute_mar(landmarks, w, h):
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])
    top = pt(INNER_TOP); bottom = pt(INNER_BOTTOM)
    left = pt(OUTER_LEFT); right = pt(OUTER_RIGHT)
    return float(np.linalg.norm(bottom - top) /
                 (np.linalg.norm(right - left) + 1e-6))


def get_lip_mar_per_frame(video_path):
    video = cv2.VideoCapture(video_path)
    fps   = video.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fm    = _get_face_mesh()
    mar_list, last = [], 0.0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        res   = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            h, w, _ = frame.shape
            m = compute_mar(res.multi_face_landmarks[0].landmark, w, h)
            last = m
        else:
            m = last
        mar_list.append(m)

    video.release()
    return np.array(mar_list, dtype=np.float32), fps, total


# ─────────────────────────────────────────────────
#  Scoring
# ─────────────────────────────────────────────────

def _normalise(a):
    s = a.std()
    return (a - a.mean()) / s if s > 1e-8 else np.zeros_like(a)


def score_lip_reader(mar, audio, fps):
    score = 0
    mar_s   = np.convolve(mar, np.ones(3)/3, mode='same')
    d_mar   = np.diff(mar_s)
    d_audio = np.diff(audio)

    # ① Delta Correlation (max 35 pts)
    # Calibrated: real=0.304 → 35pts | AI=0.289 → 22pts
    cc = np.corrcoef(_normalise(d_mar), _normalise(d_audio))[0, 1]
    dc = float(abs(cc)) if not np.isnan(cc) else 0.0
    if dc >= 0.30:
        pts = 35
    elif dc >= 0.20:
        pts = 22
    elif dc >= 0.12:
        pts = 10
    else:
        pts = 0
    score += pts
    print(f"\n   ① Delta Correlation:    |r|={dc:.3f}  ->  +{pts}/35 pts")

    # ② Voiced/Silence MAR Ratio (max 25 pts)
    # Real: 1.0-1.3 (natural) | AI: >1.4 (over-animated)
    # Calibrated: real=1.08 → 25pts | AI=1.59 → 0pts
    voiced_mask  = audio >= np.percentile(audio, 50)
    silence_mask = audio <= np.percentile(audio, 25)
    voiced_mar  = float(mar[voiced_mask].mean())  if voiced_mask.sum()  > 0 else 0.0
    silence_mar = float(mar[silence_mask].mean()) if silence_mask.sum() > 0 else 0.0
    ratio = voiced_mar / (silence_mar + 1e-6)
    if 1.00 <= ratio <= 1.30:
        pts = 25
    elif ratio < 1.00:
        pts = 15
    elif ratio <= 1.45:
        pts = 8
    else:
        pts = 0    # Over-animated → AI
    score += pts
    print(f"   ② Voiced/Silence MAR:   {voiced_mar:.3f}/{silence_mar:.3f}"
          f"  ratio={ratio:.2f}  ->  +{pts}/25 pts")

    # ③ Lip Rhythm Band Power (max 25 pts)
    # Real: moderate 0.25-0.43 | AI: over-rhythmic >0.46
    # Calibrated: real=0.40 → 25pts | AI=0.47 → 0pts
    freqs, psd = welch(mar_s, fs=fps, nperseg=min(len(mar_s), 128))
    band_mask  = (freqs >= 1.5) & (freqs <= 8.0)
    band_power = float(psd[band_mask].sum() / (psd.sum() + 1e-10))
    peak_freq  = float(freqs[band_mask][np.argmax(psd[band_mask])]) if band_mask.any() else 0.0
    if 0.22 <= band_power <= 0.43:
        pts = 25
    elif band_power < 0.22:
        pts = 12
    elif band_power <= 0.46:
        pts = 8
    else:
        pts = 0    # Over-rhythmic → AI
    score += pts
    print(f"   ③ Lip Rhythm (2-8Hz):   band_power={band_power:.2f}"
          f"  peak={peak_freq:.1f}Hz  ->  +{pts}/25 pts")

    # ④ Open/Close Velocity Asymmetry (max 15 pts)
    # Real: balanced 0.80-1.15 | AI: more asymmetric
    # Calibrated: real=1.01 → 15pts | AI=1.12 → 15pts (similar — weak signal)
    open_vel  = float(d_mar[d_mar > 0].mean()) if (d_mar > 0).any() else 0.0
    close_vel = float(abs(d_mar[d_mar < 0].mean())) if (d_mar < 0).any() else 0.0
    asym = open_vel / (close_vel + 1e-6)
    if 0.80 <= asym <= 1.15:
        pts = 15
    elif 0.65 <= asym < 0.80 or 1.15 < asym <= 1.35:
        pts = 8
    else:
        pts = 0
    score += pts
    print(f"   ④ Open/Close Asymmetry: open={open_vel:.4f}  "
          f"close={close_vel:.4f}  ratio={asym:.2f}  ->  +{pts}/15 pts")

    total = min(int(score), 100)
    print(f"\n   TOTAL: {total}/100")
    return float(total)


# ─────────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────────

def analyze_video_lips(video_path):
    print("\n" + "="*50)
    print("👄 TruthLens AI — Lip Reader v2.0")
    print("="*50)

    print("🔊 Extracting audio...")
    wav = extract_audio(video_path)
    if wav is None:
        print("⚠️  No audio — neutral score (50)")
        return 50.0, "UNKNOWN"

    print("👄 Tracking lip movements...")
    mar, fps, total = get_lip_mar_per_frame(video_path)
    if len(mar) == 0:
        print("⚠️  No face detected — neutral score (50)")
        if os.path.exists(wav): os.remove(wav)
        return 50.0, "UNKNOWN"
    print(f"✅ MAR tracked: {len(mar)} frames  "
          f"(mean={mar.mean():.3f}, CoV={mar.std()/mar.mean()*100:.1f}%)")

    print("🎵 Analysing audio energy...")
    audio = get_audio_energy_per_frame(wav, fps, len(mar))
    print(f"✅ Audio energy: mean={audio.mean():.4f}  std={audio.std():.4f}")

    print("\n📊 Lip-Audio Analysis:")
    lip_score = score_lip_reader(mar, audio, fps)
    try:
        os.remove(wav)
    except Exception:
        pass

    print(f"\n🎯 Lip Reader Score: {lip_score:.1f}/100")
    if lip_score >= 60:
        print("✅ Lip Reader: REAL — Natural speech articulation!")
        verdict = "REAL"
    elif lip_score >= 35:
        print("⚠️ Lip Reader: SUSPICIOUS — Unusual lip-audio pattern!")
        verdict = "SUSPICIOUS"
    else:
        print("❌ Lip Reader: FAKE — Over-animated / mismatched lips!")
        verdict = "FAKE"

    return lip_score, verdict


# ─────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    path = os.path.abspath(sys.argv[1] if len(sys.argv) > 1
                           else os.path.join(BASE_DIR, "test_video.mp4"))
    score, verdict = analyze_video_lips(path)
    print(f"\n🏁 Final: {verdict} ({score:.1f}/100)")
    print("="*50)
