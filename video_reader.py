import cv2

# Read the video
video = cv2.VideoCapture("test_video.mp4")

# Get video info
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"✅ Video loaded successfully!")
print(f"🎬 FPS: {fps}")
print(f"📐 Width: {width}px")
print(f"📐 Height: {height}px")
print(f"🎞️ Total Frames: {total_frames}")

# Play the video
while True:
    ret, frame = video.read()
    
    # If video ended stop
    if not ret:
        break
    
    # Resize and show frame
    resized = cv2.resize(frame, (800, 600))
    cv2.imshow("TruthLens AI - Video", resized)
    
    # Press Q to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release and close
video.release()
cv2.destroyAllWindows()

print("✅ Video finished!")