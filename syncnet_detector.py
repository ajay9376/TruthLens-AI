from syncnet_python import SyncNet
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(BASE_DIR, "test_video.mp4")

print("🔄 Loading SyncNet model...")

# Initialize SyncNet
model = SyncNet()

print("✅ SyncNet loaded!")
print("🎬 Analyzing video...")

# Analyze video
result = model.evaluate(video_path)

print(f"\n🎯 Sync Confidence: {result['confidence']:.2f}")
print(f"🎯 Sync Distance: {result['dist']:.2f}")

if result['confidence'] > 3:
    print("✅ REAL VIDEO — Lips and audio are in sync!")
else:
    print("❌ POSSIBLE DEEPFAKE — Lips and audio are out of sync!")