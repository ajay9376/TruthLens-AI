"""
TruthLens AI — Live Call Detector v1.0
=======================================
Analyzes webcam feed in real time to detect deepfakes
during video calls (Zoom, Google Meet, etc.)
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import os
import sys
import io

# Fix emoji output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── MediaPipe Setup ───
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# ─── Landmark Indices ───
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LIP_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 308
]

# ─── EAR Calculation ───
def compute_ear(landmarks, eye_indices, w, h):
    pts = [np.array([landmarks[i].x * w, landmarks[i].y * h])
           for i in eye_indices]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return float((A + B) / (2.0 * C + 1e-6))


# ─── Texture Score ───
def quick_texture_score(face_region):
    if face_region.size == 0:
        return 50.0
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    noise = gray.astype(float) - cv2.GaussianBlur(gray, (3,3), 0).astype(float)
    noise_std = np.std(noise)

    score = 0
    if laplacian_var > 100:
        score += 50
    elif laplacian_var > 50:
        score += 30
    if noise_std > 3:
        score += 50
    elif noise_std > 1.5:
        score += 25
    return min(float(score), 100)


# ─── Draw Overlay ───
def draw_overlay(frame, verdict, score, signals, face_found):
    h, w = frame.shape[:2]

    # Background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Verdict color
    if verdict == "REAL":
        color = (0, 255, 100)
    elif verdict == "DEEPFAKE":
        color = (0, 0, 255)
    else:
        color = (0, 165, 255)

    # Title
    cv2.putText(frame, "TruthLens AI", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 150, 255), 2)

    # Verdict
    cv2.putText(frame, f"{verdict}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    # Score bar
    bar_w = int(score * 2.5)
    cv2.rectangle(frame, (10, 70), (260, 85), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, 70), (10 + bar_w, 85), color, -1)
    cv2.putText(frame, f"{score:.0f}/100", (270, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Signals
    y = 105
    for name, sig_score in signals.items():
        sig_color = (0, 255, 100) if sig_score >= 60 else \
                    (0, 165, 255) if sig_score >= 40 else (0, 0, 255)
        cv2.putText(frame, f"{name}: {sig_score:.0f}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, sig_color, 1)
        y += 20

    # Face status
    if not face_found:
        cv2.putText(frame, "No face detected", (10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    # Border
    border_color = color
    cv2.rectangle(frame, (0, 0), (w-1, h-1), border_color, 3)

    return frame


# ─── Main Live Detector ───
def run_live_detector():
    print("\n" + "="*50)
    print("🎥 TruthLens AI — Live Call Detector v1.0")
    print("="*50)
    print("Press Q to quit")
    print("Press S to take screenshot")
    print("="*50)

    # Initialize
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        return

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # State variables
    blink_count    = 0
    ear_history    = []
    texture_scores = []
    in_blink       = False
    frame_count    = 0
    start_time     = time.time()

    verdict    = "ANALYZING..."
    score      = 50.0
    signals    = {
        "Texture": 50,
        "Blink":   50,
        "Lips":    50,
    }

    print("✅ Webcam opened! Analyzing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process every 2nd frame for speed
        face_found = False

        if frame_count % 2 == 0:
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                face_found = True
                lm = results.multi_face_landmarks[0].landmark

                # ── Blink Detection ──
                left_ear  = compute_ear(lm, LEFT_EYE,  w, h)
                right_ear = compute_ear(lm, RIGHT_EYE, w, h)
                ear = (left_ear + right_ear) / 2
                ear_history.append(ear)
                if len(ear_history) > 90:
                    ear_history.pop(0)

                if ear < 0.21:
                    if not in_blink:
                        blink_count += 1
                        in_blink = True
                else:
                    in_blink = False

                # ── Texture Analysis ──
                x_coords = [int(lm[i].x * w) for i in range(468)]
                y_coords = [int(lm[i].y * h) for i in range(468)]
                x_min = max(0, min(x_coords))
                x_max = min(w, max(x_coords))
                y_min = max(0, min(y_coords))
                y_max = min(h, max(y_coords))

                face_region = frame[y_min:y_max, x_min:x_max]
                tex_score   = quick_texture_score(face_region)
                texture_scores.append(tex_score)
                if len(texture_scores) > 30:
                    texture_scores.pop(0)

                # ── Lip Openness ──
                top_lip    = lm[13]
                bottom_lip = lm[14]
                lip_dist   = abs(bottom_lip.y - top_lip.y) * h

                # ── Calculate scores every 3 seconds ──
                elapsed = time.time() - start_time
                if elapsed >= 3.0 and len(ear_history) > 10:

                    # Blink score
                    blink_rate = blink_count / (elapsed / 60)
                    if 12 <= blink_rate <= 20:
                        blink_score = 85
                    elif 8 <= blink_rate <= 25:
                        blink_score = 60
                    elif blink_rate > 0:
                        blink_score = 35
                    else:
                        blink_score = 10

                    # EAR variation
                    ear_cov = np.std(ear_history) / (np.mean(ear_history) + 1e-6)
                    if ear_cov >= 0.18:
                        blink_score = min(blink_score + 15, 100)

                    # Texture score
                    avg_texture = np.mean(texture_scores) if texture_scores else 50

                    # Lip score (simple)
                    lip_score = 70 if lip_dist > 5 else 50

                    # Combined
                    signals = {
                        "Texture": round(avg_texture),
                        "Blink":   round(blink_score),
                        "Lips":    round(lip_score),
                    }

                    score = (avg_texture * 0.4 +
                             blink_score * 0.4 +
                             lip_score   * 0.2)

                    if score >= 60:
                        verdict = "REAL"
                    elif score >= 40:
                        verdict = "SUSPICIOUS"
                    else:
                        verdict = "DEEPFAKE"

        # Draw overlay
        frame = draw_overlay(frame, verdict, score, signals, face_found)

        # Show FPS
        fps = frame_count / (time.time() - start_time + 1e-6)
        cv2.putText(frame, f"FPS: {fps:.0f}", (w-80, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show frame
        cv2.imshow("TruthLens AI — Live Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_path = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(screenshot_path, frame)
            print(f"📸 Screenshot saved: {screenshot_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Live detector stopped!")
    print(f"📊 Final: {verdict} ({score:.1f}/100)")


# ─── CLI ───
if __name__ == "__main__":
    run_live_detector()