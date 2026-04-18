import cv2
import mediapipe as mp
import os

# -------- SETUP --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(BASE_DIR, "test_video.mp4")

# -------- MEDIAPIPE SETUP --------
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,  # False for video!
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------- LIP LANDMARKS --------
LIP_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 308
]

# -------- READ VIDEO --------
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("❌ Video not found!")
    exit()

print("✅ Video loaded! Press Q to quit")
frame_count = 0

while True:
    ret, frame = video.read()

    if not ret:
        break

    frame_count += 1

    # Resize frame
    frame = cv2.resize(frame, (800, 600))

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect landmarks
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Draw full face mesh in green
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=1
                )
            )

            # Draw lip points in RED
            h, w, _ = frame.shape
            for idx in LIP_LANDMARKS:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    # Show frame number
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("TruthLens AI - Lip Tracker", frame)

    # Press Q to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print(f"✅ Processed {frame_count} frames!")