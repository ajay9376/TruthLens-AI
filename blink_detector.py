"""
TruthLens AI — Blink Pattern Detector  v2.0
=============================================
Detects whether a video contains REAL or AI-generated blinking.

Calibrated from ground-truth data:
  AI video  (test_video.mp4)   — 8s, 24fps, 4 blinks
  Real video (test_video1.mp4) — 63s, 30fps, 33 blinks

Key signals that catch AI video:
  ① Blink duration too short & robotically uniform (2-3 frames every time)
     Real blinks: 2-23 frames, std ≈ 4.0
     AI blinks:   2-3 frames,  std ≈ 0.5  ← CAUGHT

  ② EAR stability too low (eyes barely close during "blinks")
     Real: min EAR can hit 0.057  (full closure)
     AI:   min EAR only hits 0.119 (partial, fake-looking closure)

  ③ Duration CoV near-zero (every blink is identical length)
     Real: Duration CoV ≈ 77%
     AI:   Duration CoV ≈ 20%  ← CAUGHT

  ④ Interval CoV too low (blinks too evenly spaced)
     Real: Interval CoV ≈ 82%
     AI:   Interval CoV ≈ 51%

  ⑤ Blink rate too high for short clip
     Normal: 12-20/min   AI video: 30/min

Score: 0-100, higher = more REAL
"""

import sys
import io
import cv2
import numpy as np
import mediapipe as mp
import os

# Fix emoji output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────────
#  MediaPipe setup (lazy init)
# ──────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
_face_mesh   = None

def _get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return _face_mesh


# Eye landmark indices (MediaPipe 468-point model)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# EAR below this → eye closing / closed
EAR_THRESHOLD    = 0.21
# Minimum consecutive frames below threshold to count as a blink
MIN_BLINK_FRAMES = 2


# ──────────────────────────────────────────────────
#  EAR calculation
# ──────────────────────────────────────────────────

def compute_ear(landmarks, eye_indices, w, h):
    pts = [np.array([landmarks[i].x * w, landmarks[i].y * h])
           for i in eye_indices]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return float((A + B) / (2.0 * C + 1e-6))


# ──────────────────────────────────────────────────
#  Blink extraction
# ──────────────────────────────────────────────────

def extract_blinks(ear_series):
    """
    Walk the EAR time-series and return a list of blink events.
    Each event: {'start': frame_idx, 'duration': n_frames, 'min_ear': float}
    """
    blinks    = []
    in_blink  = False
    start     = 0
    run       = 0
    run_min   = 1.0

    for i, ear in enumerate(ear_series):
        if ear < EAR_THRESHOLD:
            if not in_blink:
                start    = i
                in_blink = True
                run_min  = ear
            run    += 1
            run_min = min(run_min, ear)
        else:
            if in_blink:
                if run >= MIN_BLINK_FRAMES:
                    blinks.append({
                        'start':    start,
                        'duration': run,
                        'min_ear':  run_min,
                    })
                in_blink = False
                run      = 0
                run_min  = 1.0

    return blinks


# ──────────────────────────────────────────────────
#  Scoring
# ──────────────────────────────────────────────────

def score_blink_pattern(blinks, ear_series, fps, total_frames):
    """
    Score 0-100 (higher = more REAL) based on five signals.

    Calibrated thresholds:
      AI video  → expect score ≤ 30  → FAKE
      Real video → expect score ≥ 65  → REAL
    """
    duration_sec = total_frames / fps
    n            = len(blinks)
    ears         = np.array(ear_series)

    # ── Pre-compute stats ──────────────────────────
    blink_rate = n / (duration_sec / 60.0) if n > 0 else 0.0

    durations  = np.array([b['duration'] for b in blinks]) if n > 0 else np.array([])
    min_ears   = np.array([b['min_ear']  for b in blinks]) if n > 0 else np.array([])

    if n >= 2:
        starts      = np.array([b['start'] for b in blinks])
        intervals   = np.diff(starts) / fps          # seconds between blinks
        ivl_mean    = float(np.mean(intervals))
        ivl_std     = float(np.std(intervals))
        ivl_cov     = ivl_std / (ivl_mean + 1e-6)   # CoV as fraction

        dur_mean    = float(np.mean(durations))
        dur_std     = float(np.std(durations))
        dur_cov     = dur_std / (dur_mean + 1e-6)

        avg_min_ear = float(np.mean(min_ears))
    else:
        ivl_mean = ivl_std = ivl_cov = 0.0
        dur_mean = dur_std = dur_cov = 0.0
        avg_min_ear = float(np.mean(min_ears)) if n == 1 else EAR_THRESHOLD

    ear_cov = float(ears.std() / (ears.mean() + 1e-6))  # overall EAR variability

    # ── Print diagnostics ──────────────────────────
    print(f"\n📊 Blink Analysis (calibrated v2):")
    print(f"   Duration          : {duration_sec:.1f} s  |  FPS: {fps:.0f}")
    print(f"   Blinks detected   : {n}")
    print(f"   Blink rate        : {blink_rate:.1f} /min  (normal: 12–20)")
    if n >= 2:
        print(f"   Interval mean/std : {ivl_mean:.2f}s / {ivl_std:.2f}s  CoV={ivl_cov*100:.1f}%"
              f"  (real≥60%, AI<55%)")
        print(f"   Duration mean/std : {dur_mean:.1f}f / {dur_std:.1f}f  CoV={dur_cov*100:.1f}%"
              f"  (real≥50%, AI<30%) ← KEY SIGNAL")
        print(f"   Avg min EAR       : {avg_min_ear:.3f}"
              f"  (real<0.18, AI>0.10 but rarely deep)")
    print(f"   Overall EAR CoV   : {ear_cov*100:.1f}%  (real≥18%, AI<17%)")

    score = 0

    # ══════════════════════════════════════════════
    # ① BLINK DURATION VARIATION  (max 30 pts)
    #   Most powerful signal — AI blinks are robotically uniform
    #   Real: dur_cov ≈ 77%   |   AI: dur_cov ≈ 20%
    # ══════════════════════════════════════════════
    if n >= 2:
        if dur_cov >= 0.60:
            pts = 30          # Natural variation (77% real baseline)
        elif dur_cov >= 0.40:
            pts = 18
        elif dur_cov >= 0.20:
            pts = 8           # Borderline — AI sits right at 20%
        else:
            pts = 0           # Robot-identical blinks → AI
        score += pts
        print(f"\n   ① Duration CoV {dur_cov*100:.1f}%  →  +{pts}/30 pts")
    else:
        score += 0
        print(f"\n   ① Duration CoV  N/A (< 2 blinks)  →  +0/30 pts")

    # ══════════════════════════════════════════════
    # ② EYE CLOSURE DEPTH  (max 25 pts)
    #   Real blinks reach very low EAR (full closure)
    #   AI blinks stay shallow (min EAR stays high)
    #   Real: avg_min_ear ≈ 0.10–0.15  |  AI: avg_min_ear ≈ 0.12–0.16
    #   Better signal: std of min_ears (real varies, AI is uniform)
    # ══════════════════════════════════════════════
    if n >= 1:
        min_ear_std = float(np.std(min_ears)) if n > 1 else 0.0
        if avg_min_ear < 0.13 and min_ear_std > 0.02:
            pts = 25          # Deep, varied closure → REAL
        elif avg_min_ear < 0.16:
            pts = 14
        elif avg_min_ear < 0.19:
            pts = 5
        else:
            pts = 0           # Never fully closes → AI
        score += pts
        print(f"   ② Min EAR {avg_min_ear:.3f}  std={min_ear_std:.3f}  →  +{pts}/25 pts")
    else:
        score += 0
        print(f"   ② Min EAR  N/A  →  +0/25 pts")

    # ══════════════════════════════════════════════
    # ③ INTERVAL IRREGULARITY  (max 20 pts)
    #   Real blinks are irregular (CoV ≈ 82%)
    #   AI blinks are semi-regular (CoV ≈ 51%)
    # ══════════════════════════════════════════════
    if n >= 2:
        if ivl_cov >= 0.75:
            pts = 20          # Highly irregular → natural
        elif ivl_cov >= 0.55:
            pts = 12
        elif ivl_cov >= 0.40:
            pts = 5           # AI sits around 51%
        else:
            pts = 0           # Too regular → AI
        score += pts
        print(f"   ③ Interval CoV {ivl_cov*100:.1f}%  →  +{pts}/20 pts")
    else:
        score += 0
        print(f"   ③ Interval CoV  N/A  →  +0/20 pts")

    # ══════════════════════════════════════════════
    # ④ BLINK RATE  (max 15 pts)
    #   Normal human: 12-20 blinks/min
    #   AI: often 0 (no blinks) or > 25 (over-compensating)
    # ══════════════════════════════════════════════
    if 12 <= blink_rate <= 20:
        pts = 15
    elif 8 <= blink_rate < 12 or 20 < blink_rate <= 24:
        pts = 8
    elif 5 <= blink_rate < 8 or 24 < blink_rate <= 30:
        pts = 3               # AI video: 30/min — just barely scores
    else:
        pts = 0               # 0 blinks or wildly high → AI
    score += pts
    print(f"   ④ Blink rate {blink_rate:.1f}/min  →  +{pts}/15 pts")

    # ══════════════════════════════════════════════
    # ⑤ OVERALL EAR VARIABILITY  (max 10 pts)
    #   Real faces: eyes move and vary naturally (CoV ≥ 18%)
    #   AI faces: eyes are more static between blinks (CoV < 17%)
    # ══════════════════════════════════════════════
    if ear_cov >= 0.20:
        pts = 10
    elif ear_cov >= 0.17:
        pts = 5
    else:
        pts = 0
    score += pts
    print(f"   ⑤ EAR CoV {ear_cov*100:.1f}%  →  +{pts}/10 pts")

    total = min(int(score), 100)
    print(f"\n   TOTAL: {total}/100")
    return float(total)


# ──────────────────────────────────────────────────
#  Main pipeline
# ──────────────────────────────────────────────────

def analyze_video_blinks(video_path):
    """
    Run blink analysis on a video file.
    Returns (score: float, verdict: str).
    """
    print("\n" + "="*50)
    print("👁️  TruthLens AI — Blink Detector v2.0")
    print("="*50)

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"❌ Cannot open: {video_path}")
        return 50.0, "UNKNOWN"

    fps          = video.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print("❌ Video has 0 frames!")
        video.release()
        return 50.0, "UNKNOWN"

    face_mesh  = _get_face_mesh()
    ear_series = []
    frame_no   = 0
    faces_found = 0

    print(f"🎬 Processing {total_frames} frames at {fps:.0f} fps...")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame   = cv2.resize(frame, (640, 480))
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            faces_found += 1
            lm        = results.multi_face_landmarks[0].landmark
            h, w, _   = frame.shape
            left_ear  = compute_ear(lm, LEFT_EYE,  w, h)
            right_ear = compute_ear(lm, RIGHT_EYE, w, h)
            ear_series.append((left_ear + right_ear) / 2.0)
        else:
            ear_series.append(ear_series[-1] if ear_series else 0.30)

        frame_no += 1

    video.release()
    print(f"✅ Faces in {faces_found}/{total_frames} frames")

    # Extract blink events
    blinks = extract_blinks(ear_series)

    # Score
    blink_score = score_blink_pattern(blinks, ear_series, fps, total_frames)

    print(f"\n🎯 Blink Score: {blink_score:.1f}/100")

    if blink_score >= 65:
        print("✅ Blink: REAL — Natural eye-blink pattern!")
        verdict = "REAL"
    elif blink_score >= 35:
        print("⚠️ Blink: SUSPICIOUS — Unusual blink pattern!")
        verdict = "SUSPICIOUS"
    else:
        print("❌ Blink: FAKE — Unnatural / AI-generated blink pattern!")
        verdict = "FAKE"

    return blink_score, verdict


# ──────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys
    path = _sys.argv[1] if len(_sys.argv) > 1 else os.path.join(BASE_DIR, "test_video.mp4")
    score, verdict = analyze_video_blinks(path)
    print(f"\n🏁 Final: {verdict} ({score:.1f}/100)")
    print("="*50)
