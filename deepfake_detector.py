import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYNCNET_DIR = os.path.join(BASE_DIR, "syncnet_python")
FFMPEG_PATH = r"C:\Users\gujju\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin"

def add_ffmpeg_to_path():
    os.environ["PATH"] += os.pathsep + FFMPEG_PATH

def check_audio_stream(video_path):
    """Returns True if video has an audio stream, False otherwise."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type",
         "-of", "default=noprint_wrappers=1", video_path],
        capture_output=True, text=True
    )
    return "codec_type=audio" in result.stdout

def convert_video(input_path, output_path):
    print("🎬 Converting video for analysis...")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", "transpose=2,scale=224:224",  # ← rotation fix added back!
        "-r", "25",
        "-ac", "1",
        "-ar", "16000",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ ffmpeg ERROR:")
        print(result.stderr[-2000:])
    else:
        print("✅ Video converted!")

def run_syncnet(video_path):
    print("🤖 Running SyncNet analysis...")
    result = subprocess.run(
        [sys.executable, "demo_syncnet.py",
         "--videofile", video_path,
         "--tmp_dir", "tmp_truthlens"],
        capture_output=True,
        text=True,
        cwd=SYNCNET_DIR
    )
    output = result.stdout + result.stderr
    print("\n🔧 RAW SYNCNET OUTPUT (debug):")
    print("-" * 40)
    print(output[-3000:] if len(output) > 3000 else output)
    print("-" * 40)
    return output

def parse_results(output):
    confidence = None
    min_dist = None
    av_offset = None

    for line in output.split('\n'):
        if 'Confidence:' in line:
            confidence = float(line.split(':')[1].strip())
        if 'Min dist:' in line:
            min_dist = float(line.split(':')[1].strip())
        if 'AV offset:' in line:
            av_offset = float(line.split(':')[1].strip())

    return confidence, min_dist, av_offset

def detect_deepfake(video_path):
    print("\n" + "="*50)
    print("🔍 TruthLens AI — Deepfake Detector")
    print("="*50)

    # Add ffmpeg to path
    add_ffmpeg_to_path()

    # Check if video has audio
    if not check_audio_stream(video_path):
        print("\n❌ ERROR: This video has NO audio track!")
        print("SyncNet needs a video with speech audio to detect lip sync.")
        print("Please use a video where someone is speaking.")
        print("="*50)
        return

    # Convert video
    converted_path = os.path.join(
        SYNCNET_DIR, "data", "input_video.avi"
    )
    convert_video(video_path, converted_path)

    # Run SyncNet
    output = run_syncnet("data/input_video.avi")

    # Parse results
    confidence, min_dist, av_offset = parse_results(output)

    print("\n" + "="*50)
    print("📊 ANALYSIS RESULTS")
    print("="*50)

    if confidence is not None:
        print(f"🎯 Confidence Score: {confidence:.3f}")
        print(f"📏 Min Distance: {min_dist:.3f}")
        print(f"⏱️ AV Offset: {av_offset}")

        if confidence > 1.5 and (av_offset is not None and av_offset <= 1):
            print("\n✅ VERDICT: REAL VIDEO")
            print("Lips and audio are in sync!")
        elif confidence > 0.5 or (av_offset is not None and av_offset <= 2):
            print("\n⚠️ VERDICT: SUSPICIOUS / INCONCLUSIVE")
            print("Could not confidently confirm sync — manual review recommended.")
        else:
            print("\n❌ VERDICT: POSSIBLE DEEPFAKE")
            print("Lips and audio appear out of sync!")
    else:
        print("❌ Could not analyze video!")
        print("Make sure face is clearly visible!")

    print("="*50)

# -------- RUN --------
if __name__ == "__main__":
    video_path = os.path.join(BASE_DIR, "test_video.mp4")
    detect_deepfake(video_path)