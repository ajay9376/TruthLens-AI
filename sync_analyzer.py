import cv2
import librosa
import numpy as np
import mediapipe as mp
import os
from scipy.ndimage import uniform_filter1d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(BASE_DIR, "test_video.mp4")
audio_path = os.path.join(BASE_DIR, "extracted_audio.wav")

# -------- LOAD AUDIO --------
print("🔊 Loading audio...")
y, sr = librosa.load(audio_path)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
print(f"✅ Audio loaded! MFCC shape: {mfcc.shape}")

# -------- LOAD VIDEO --------
print("🎬 Loading video...")
video = cv2.VideoCapture(video_path)
fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"✅ Video loaded! FPS: {fps} Total frames: {total_frames}")

# -------- MEDIAPIPE SETUP --------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LIP_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 308
]

# -------- SYNC AUDIO + VIDEO --------
print("🔄 Syncing audio and video...")

frame_count = 0
lip_movements = []
audio_energies = []

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.resize(frame, (800, 600))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get matching audio frame
    audio_frame_idx = int((frame_count / total_frames) * mfcc.shape[1])
    audio_frame_idx = min(audio_frame_idx, mfcc.shape[1] - 1)

    # Get audio energy at this moment
    audio_energy = np.mean(np.abs(mfcc[:, audio_frame_idx]))
    audio_energies.append(audio_energy)

    # Get lip movement
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            lip_points = []
            for idx in LIP_LANDMARKS:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y_coord = int(landmark.y * h)
                lip_points.append((x, y_coord))

            # Calculate lip openness
            top_lip = lip_points[0][1]
            bottom_lip = lip_points[11][1]
            lip_distance = abs(bottom_lip - top_lip)
            lip_movements.append(lip_distance)
    else:
        lip_movements.append(0)

video.release()

# -------- ANALYZE SYNC --------
print(f"✅ Processed {frame_count} frames!")
print(f"👄 Lip movements recorded: {len(lip_movements)}")
print(f"🔊 Audio energies recorded: {len(audio_energies)}")

# Normalize both
lip_norm = np.array(lip_movements) / (max(lip_movements) + 1e-6)
audio_norm = np.array(audio_energies) / (max(audio_energies) + 1e-6)

# Calculate multiple correlation methods
correlation1 = np.corrcoef(lip_norm, audio_norm)[0, 1]

# Smoothed version
lip_smooth = uniform_filter1d(lip_norm, size=10)
audio_smooth = uniform_filter1d(audio_norm, size=10)
correlation2 = np.corrcoef(lip_smooth, audio_smooth)[0, 1]

# Take best correlation
best_correlation = max(correlation1, correlation2)
sync_score = (best_correlation + 1) / 2 * 100

print(f"\n🎯 Raw Sync Score: {(correlation1 + 1) / 2 * 100:.2f}%")
print(f"🎯 Smoothed Sync Score: {(correlation2 + 1) / 2 * 100:.2f}%")
print(f"🎯 Final Sync Score: {sync_score:.2f}%")

if sync_score > 50:
    print("✅ REAL VIDEO — Lips and audio are in sync!")
else:
    print("❌ POSSIBLE DEEPFAKE — Lips and audio are out of sync!")