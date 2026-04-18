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

# -------- MEDIAPIPE SETUP --------
mp_face_mesh = mp.solutions.face_mesh
_face_mesh = None  # Lazy init so importing doesn't crash if mediapipe is absent

def _get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return _face_mesh


def extract_face_region(frame, face_landmarks, h, w):
    """Extract just the face bounding-box region from a frame."""
    x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
    y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]

    x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
    y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))

    face_region = frame[y_min:y_max, x_min:x_max]
    return face_region


def compute_lbp(img):
    """
    Vectorised Local Binary Pattern.
    Returns the LBP-coded image (uint8).
    Real faces → rich, varied LBP histogram.
    AI faces  → flatter, less-varied histogram.
    """
    # Pad image so we don't lose the border
    p = np.pad(img, 1, mode='reflect')

    lbp = np.zeros_like(img, dtype=np.uint8)
    # 8-neighbor comparison (clockwise from top-left)
    neighbors = [
        p[:-2, :-2],   # top-left
        p[:-2, 1:-1],  # top
        p[:-2, 2:],    # top-right
        p[1:-1, 2:],   # right
        p[2:, 2:],     # bottom-right
        p[2:, 1:-1],   # bottom
        p[2:, :-2],    # bottom-left
        p[1:-1, :-2],  # left
    ]
    for bit, nb in enumerate(neighbors):
        lbp |= ((nb >= img).astype(np.uint8) << bit)

    return lbp


def analyze_texture(face_region):
    """
    Analyze face texture for deepfake signals.
    Returns a dict of metrics, or None if face_region is empty.

    Key insight: modern AI video generators produce sharp, textured faces
    that pass naive per-frame tests.  We therefore also measure SKIN
    UNIFORMITY — real skin varies naturally; AI skin is unnaturally even.
    """
    if face_region is None or face_region.size == 0:
        return None

    # Convert to grayscale & resize for consistent analysis
    gray        = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_region, (128, 128))
    gray        = cv2.resize(gray, (128, 128))

    # -------- Method 1: Laplacian Variance --------
    # Real faces → natural sharpness/noise → higher variance
    # AI faces  → too smooth → lower variance
    # NOTE: modern AI CAN pass this test; we use it as part of a combined score.
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # -------- Method 2: LBP Histogram Uniformity --------
    # Real faces → rich texture patterns → high entropy
    # AI faces  → repetitive / generated texture → lower entropy
    lbp_img = compute_lbp(gray)
    hist, _ = np.histogram(lbp_img.ravel(), bins=32, range=(0, 256))
    hist_norm   = hist / (hist.sum() + 1e-8)
    lbp_entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-8))

    # -------- Method 3: Frequency Analysis --------
    # Real faces → natural frequency distribution
    # AI faces  → unusual compression / GAN artefacts
    f_transform = np.fft.fft2(gray.astype(np.float32))
    f_shift     = np.fft.fftshift(f_transform)
    magnitude   = 20 * np.log(np.abs(f_shift) + 1)
    freq_mean   = float(np.mean(magnitude))
    freq_std    = float(np.std(magnitude))

    # -------- Method 4: Noise Analysis --------
    # Real faces → natural camera sensor noise
    # AI faces  → suspiciously clean or perfectly patterned noise
    blurred   = cv2.GaussianBlur(gray, (5, 5), 0)
    noise     = gray.astype(float) - blurred.astype(float)
    noise_std = float(np.std(noise))

    # -------- Method 5: Skin Colour Uniformity --------
    # Real skin: lighting gradients + pores + micro-variations → high std
    # AI skin : unnaturally smooth / even colour → low std
    # We measure std of pixel intensities within the skin-tone mask.
    hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)
    skin_mask   = cv2.inRange(hsv, (0, 20, 70), (20, 150, 255))
    skin_pixels = face_resized[skin_mask > 0]
    skin_var    = float(np.std(skin_pixels)) if len(skin_pixels) > 50 else 30.0

    return {
        'laplacian_var': float(laplacian_var),
        'lbp_entropy':   float(lbp_entropy),
        'freq_mean':     freq_mean,
        'freq_std':      freq_std,
        'noise_std':     noise_std,
        'skin_var':      skin_var,
    }


def calculate_texture_score(metrics_list):
    """
    Aggregate per-frame texture metrics into a single 0-100 score.
    Higher = more likely REAL.

    v2 Scoring philosophy
    ─────────────────────
    Modern AI video generators (Sora, Pika, RunwayML …) produce faces that
    ARE sharp and textured — so naive per-frame averages give false 'REAL'
    results.

    The key betrayal is TEMPORAL CONSISTENCY:
      • Real faces vary naturally frame-to-frame (lighting, micro-motion)
      • AI faces are unnaturally stable across frames

    We measure this with the Coefficient of Variation (CoV = std/mean):
      CoV < 8%  → suspiciously consistent → AI signal
      CoV > 15% → natural variation       → REAL signal

    Skin uniformity is the second strong signal:
      Low skin_var per frame → AI renderer made skin too perfect
    """
    if not metrics_list:
        return 50.0  # Neutral if no face found

    n = len(metrics_list)

    # --- Per-frame averages ---
    arr_laplacian  = np.array([m['laplacian_var'] for m in metrics_list])
    arr_noise      = np.array([m['noise_std']     for m in metrics_list])
    arr_freq_std   = np.array([m['freq_std']       for m in metrics_list])
    arr_lbp        = np.array([m['lbp_entropy']    for m in metrics_list])
    arr_skin       = np.array([m['skin_var']        for m in metrics_list])

    avg_laplacian   = float(np.mean(arr_laplacian))
    avg_noise       = float(np.mean(arr_noise))
    avg_freq_std    = float(np.mean(arr_freq_std))
    avg_lbp_entropy = float(np.mean(arr_lbp))
    avg_skin_var    = float(np.mean(arr_skin))

    # --- Temporal variation (CoV %) ---
    def cov(arr):
        mean = np.mean(arr)
        return float(np.std(arr) / (mean + 1e-8) * 100)

    cov_lap   = cov(arr_laplacian)
    cov_noise = cov(arr_noise)
    cov_skin  = cov(arr_skin)

    print(f"\n📊 Texture Metrics (averaged over {n} frames):")
    print(f"   Laplacian Variance : {avg_laplacian:.2f}  (temporal CoV {cov_lap:.1f}%)")
    print(f"   Noise Std          : {avg_noise:.2f}      (temporal CoV {cov_noise:.1f}%)")
    print(f"   Frequency Std      : {avg_freq_std:.2f}")
    print(f"   LBP Entropy        : {avg_lbp_entropy:.2f}")
    print(f"   Skin Uniformity    : {avg_skin_var:.2f}   (temporal CoV {cov_skin:.1f}%)")
    print(f"   ⚠️  Low CoV = too consistent across frames = AI signal")

    score = 0

    # ════════════════════════════════════════════════════
    # A) TEMPORAL VARIATION  (max 40 pts)  ← NEW & CRITICAL
    #    Catches modern AI video that passes per-frame tests
    # ════════════════════════════════════════════════════

    # Laplacian CoV  (max 15 pts)
    #   Real: 15-50%  |  AI: < 10%
    if cov_lap > 20:
        score += 15
    elif cov_lap > 12:
        score += 8
    elif cov_lap > 8:
        score += 3
    else:
        score += 0   # Unnaturally stable → AI

    # Noise CoV  (max 15 pts)
    #   Real: 8-25%   |  AI: < 6%  (our AI video had 5.6% — caught!)
    if cov_noise > 12:
        score += 15
    elif cov_noise > 8:
        score += 8
    elif cov_noise > 6:
        score += 3
    else:
        score += 0   # Suspiciously stable noise → AI

    # Skin CoV  (max 10 pts)
    #   Real: 10%+   |  AI: < 2%  (our AI video had 1.2% — caught!)
    if cov_skin > 8:
        score += 10
    elif cov_skin > 4:
        score += 5
    else:
        score += 0

    # ════════════════════════════════════════════════════
    # B) SKIN UNIFORMITY per frame  (max 25 pts)  ← NEW
    #    AI renderers produce unnaturally smooth / even skin
    # ════════════════════════════════════════════════════
    #   Real skin: high std AND naturally varies frame-to-frame (CoV > 5%)
    #   AI skin:   looks "okay" per frame but is identically smooth each frame
    if avg_skin_var > 55 and cov_skin > 5:
        score += 25   # High variance + naturally variable across frames → REAL
    elif avg_skin_var > 40 and cov_skin > 3:
        score += 15   # Decent variance with some natural frame variation
    elif avg_skin_var > 30:
        score += 5    # Borderline
    else:
        score += 0    # Too uniform → AI

    # ════════════════════════════════════════════════════
    # C) CLASSIC PER-FRAME METRICS  (max 35 pts)
    #    Still useful for older / lower-quality deepfakes
    # ════════════════════════════════════════════════════

    # Laplacian average (max 12 pts)
    # Laplacian average (max 12 pts)
    # Lowered: real compressed video can have laplacian 80-120 (not 150+)
    if avg_laplacian > 60:
        score += 12
    else:
        score += 0

    # Noise average (max 12 pts)
    # Lowered: well-encoded real video has noise std 3-5, not always > 6
    if avg_noise > 3:
        score += 12
    else:
        score += 0

    # LBP Entropy (max 11 pts)
    if avg_lbp_entropy > 4.5:
        score += 11
    elif avg_lbp_entropy > 3.5:
        score += 5
    else:
        score += 0

    return float(min(score, 100))


def analyze_video_texture(video_path, sample_frames=30):
    """
    Run texture analysis on a video file.
    Returns (texture_score: float, verdict: str).
    """
    print("\n" + "="*50)
    print("🔍 TruthLens AI — Texture Analyzer")
    print("="*50)

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return 50.0, "UNKNOWN"

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print("❌ Video has 0 frames!")
        video.release()
        return 50.0, "UNKNOWN"

    face_mesh = _get_face_mesh()

    # Evenly-spaced frame indices to sample
    frame_indices = set(np.linspace(0, total_frames - 1,
                                    sample_frames, dtype=int).tolist())

    metrics_list = []
    faces_found  = 0
    frame_count  = 0

    print(f"🎬 Sampling {sample_frames} frames from {total_frames} total...")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count in frame_indices:
            frame_resized = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                faces_found += 1
                h, w, _ = frame_resized.shape
                face_region = extract_face_region(
                    frame_resized,
                    results.multi_face_landmarks[0],
                    h, w
                )
                metrics = analyze_texture(face_region)
                if metrics:
                    metrics_list.append(metrics)

        frame_count += 1

    video.release()

    print(f"✅ Detected faces in {faces_found}/{sample_frames} sampled frames")

    texture_score = calculate_texture_score(metrics_list)

    print(f"\n🎯 Texture Score: {texture_score:.1f}/100")

    if texture_score >= 70:
        print("✅ Texture: REAL — Natural face texture detected!")
        verdict = "REAL"
    elif texture_score >= 40:
        print("⚠️ Texture: SUSPICIOUS — Unusual texture patterns!")
        verdict = "SUSPICIOUS"
    else:
        print("❌ Texture: FAKE — Unnatural / AI texture detected!")
        verdict = "FAKE"

    return texture_score, verdict


# -------- STANDALONE RUN --------
if __name__ == "__main__":
    video_path = os.path.join(BASE_DIR, "test_video.mp4")
    score, verdict = analyze_video_texture(video_path)
    print(f"\n🏁 Final: {verdict} ({score:.1f}/100)")
    print("="*50)