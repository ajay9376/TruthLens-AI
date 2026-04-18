import cv2
import mediapipe as mp
import os

# -------- FIX PATH ISSUE --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "image_test.jpg")

print("Looking for image at:")
print(image_path)

image = cv2.imread(image_path)

if image is None:
    print("❌ Image not found!")
    exit()
else:
    print("✅ Image loaded successfully!")

# -------- MEDIAPIPE SETUP --------
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=2,
    min_detection_confidence=0.3
)

# -------- LIP LANDMARK INDICES --------
LIP_LANDMARKS = [
    # Upper lip
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    # Lower lip
    146, 91, 181, 84, 17, 314, 405, 321, 375,
    # Lip corners
    78, 308
]

# -------- RESIZE FOR BETTER DETECTION --------
image = cv2.resize(image, (1920, 1080))

# -------- CONVERT BGR → RGB --------
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# -------- DETECT LANDMARKS --------
results = face_mesh.process(rgb)

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:

        # Draw all 468 points (green mesh)
        mp_draw.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_draw.DrawingSpec(
                color=(0, 255, 0),
                thickness=1
            )
        )

        # Draw ONLY lip points in RED on top
        h, w, _ = image.shape
        for idx in LIP_LANDMARKS:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    print("✅ Face landmarks detected!")
    print(f"👄 Lip points highlighted in RED: {len(LIP_LANDMARKS)}")

else:
    print("❌ No face detected!")

# -------- SHOW RESULT --------
resized = cv2.resize(image, (800, 600))
cv2.imshow("TruthLens AI - Lip Landmarks", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()