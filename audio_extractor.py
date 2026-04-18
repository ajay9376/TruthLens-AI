from moviepy import VideoFileClip
import os

# -------- SETUP --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(BASE_DIR, "test_video.mp4")
audio_path = os.path.join(BASE_DIR, "extracted_audio.wav")

print("🎬 Loading video...")

# -------- LOAD VIDEO --------
video = VideoFileClip(video_path)

# -------- EXTRACT AUDIO --------
print("🔊 Extracting audio...")
video.audio.write_audiofile(audio_path)

print("✅ Audio extracted successfully!")
print(f"📁 Saved as: extracted_audio.wav")
print(f"⏱️ Duration: {video.duration} seconds")
print(f"🎵 Audio FPS: {video.audio.fps}")