from ultralytics import YOLO
import cv2

# Load YOLO face detection model
model = YOLO("yolov8n.pt")

# Read image
image = cv2.imread("test_image.jpg")

# Detect faces
results = model(image, conf=0.5) 

# Draw boxes
for result in results:
    boxes = result.boxes
    print(f"✅ Faces detected: {len(boxes)}")
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Show result
resized = cv2.resize(image, (800, 600))
cv2.imshow("TruthLens AI - Face Detection V2", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()