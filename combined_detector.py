"""
TruthLens AI — Combined Deepfake Detector
==========================================
Merges three independent signals into one confidence score:

  Signal              Weight   Source
  ─────────────────────────────────────────
  Face Texture         34%    texture_analyzer.py
  Blink Pattern        33%    blink_detector.py
  Lip Reader           33%    lip_reader.py

  SyncNet: Available locally, disabled on cloud

Usage:
    python combined_detector.py                     # uses test_video.mp4
    python combined_detector.py my_video.mp4
"""

import os
import sys
import io
import subprocess
import numpy as np

# Fix emoji output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FFMPEG_PATH = r"C:\Users\gujju\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin"

# Detect if running on cloud or locally
IS_CLOUD = os.name != 'nt'  # True on Linux (cloud), False on Windows (local)

# ─────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────

def _banner(title: str):
    width = 56
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def _section(title: str):
    print(f"\n{'─'*48}")
    print(f"  {title}")
    print(f"{'─'*48}")


def add_ffmpeg_to_path():
    if IS_CLOUD:
        os.environ["PATH"] += ":/usr/bin:/usr/local/bin"
    else:
        if FFMPEG_PATH not in os.environ.get("PATH", ""):
            os.environ["PATH"] += os.pathsep + FFMPEG_PATH


# ─────────────────────────────────────────────────
#  1. SyncNet Score
# ─────────────────────────────────────────────────

SYNCNET_DIR = os.path.join(BASE_DIR, "syncnet_python")


def _check_audio(video_path: str) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type",
         "-of", "default=noprint_wrappers=1", video_path],
        capture_output=True, text=True
    )
    return "codec_type=audio" in result.stdout


def _convert_video(input_path: str, output_path: str):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", "transpose=2,scale=224:224",
        "-r", "25",
        "-ac", "1",
        "-ar", "16000",
        output_path
    ]
    subprocess.run(cmd, capture_output=True, text=True)


def _run_syncnet(video_path: str) -> str:
    result = subprocess.run(
        [sys.executable, "demo_syncnet.py",
         "--videofile", video_path,
         "--tmp_dir", "tmp_truthlens"],
        capture_output=True,
        text=True,
        cwd=SYNCNET_DIR
    )
    return result.stdout + result.stderr


def _parse_syncnet(output: str):
    confidence = min_dist = av_offset = None
    for line in output.split('\n'):
        if 'Confidence:' in line:
            try:
                confidence = float(line.split(':')[1].strip())
            except ValueError:
                pass
        if 'Min dist:' in line:
            try:
                min_dist = float(line.split(':')[1].strip())
            except ValueError:
                pass
        if 'AV offset:' in line:
            try:
                av_offset = float(line.split(':')[1].strip())
            except ValueError:
                pass
    return confidence, min_dist, av_offset


def get_syncnet_score(video_path: str) -> float:
    _section("① SyncNet Lip-Sync Analysis")

    # Disable on cloud
    if IS_CLOUD:
        print("⚠️  SyncNet disabled on cloud — 3-signal mode active")
        return None

    if not _check_audio(video_path):
        print("⚠️  No audio track found — skipping SyncNet")
        return None

    converted = os.path.join(SYNCNET_DIR, "data", "input_video.avi")
    print("🎬 Converting video …")
    _convert_video(video_path, converted)

    print("🤖 Running SyncNet …")
    raw_output = _run_syncnet("data/input_video.avi")

    confidence, min_dist, av_offset = _parse_syncnet(raw_output)

    if confidence is None:
        print("⚠️  SyncNet returned no confidence value")
        return None

    print(f"   Confidence : {confidence:.3f}")
    print(f"   Min Dist   : {min_dist:.3f}")
    print(f"   AV Offset  : {av_offset}")

    if confidence > 1.5 and (av_offset is not None and abs(av_offset) <= 1):
        score = 85.0 + min(confidence - 1.5, 1.5) * 10
    elif confidence > 0.5 or (av_offset is not None and abs(av_offset) <= 2):
        score = 40.0 + (confidence - 0.5) * 22.5
    else:
        score = max(0.0, confidence * 40.0)

    score = float(np.clip(score, 0, 100))
    print(f"   → SyncNet Score : {score:.1f}/100")
    return score


# ─────────────────────────────────────────────────
#  2. Texture Score
# ─────────────────────────────────────────────────

def get_texture_score(video_path: str) -> float:
    _section("② Face Texture Analysis")
    try:
        from texture_analyzer import analyze_video_texture
        score, verdict = analyze_video_texture(video_path, sample_frames=30)
        print(f"   → Texture Score : {score:.1f}/100  [{verdict}]")
        return float(score)
    except Exception as exc:
        print(f"⚠️  Texture analysis failed: {exc}")
        return None


# ─────────────────────────────────────────────────
#  3. Blink Score
# ─────────────────────────────────────────────────

def get_blink_score(video_path: str) -> float:
    _section("③ Blink Pattern Detection")
    try:
        from blink_detector import analyze_video_blinks
        score, verdict = analyze_video_blinks(video_path)
        print(f"   → Blink Score : {score:.1f}/100  [{verdict}]")
        return float(score)
    except Exception as exc:
        print(f"⚠️  Blink analysis failed: {exc}")
        return None


# ─────────────────────────────────────────────────
#  4. Lip Reader Score
# ─────────────────────────────────────────────────

def get_lip_score(video_path: str) -> float:
    _section("④ Lip Reader — Speech Articulation")
    try:
        from lip_reader import analyze_video_lips
        score, verdict = analyze_video_lips(video_path)
        print(f"   → Lip Reader Score : {score:.1f}/100  [{verdict}]")
        return float(score)
    except Exception as exc:
        print(f"⚠️  Lip analysis failed: {exc}")
        return None


# ─────────────────────────────────────────────────
#  Combined Score & Verdict
# ─────────────────────────────────────────────────

def combine_scores(syncnet, texture, blink, lip) -> float:
    """
    Dynamically weight available signals.
    If a signal is None it is excluded and weights are redistributed.
    """
    signals = {
        'syncnet':    syncnet,
        'texture':    texture,
        'blink':      blink,
        'lip_reader': lip,
    }

    base_weights = {
        'syncnet':    0.20,
        'texture':    0.20,
        'blink':      0.40,
        'lip_reader': 0.20,
    }

    # Filter out None signals
    active = {k: v for k, v in signals.items() if v is not None}

    if not active:
        return 50.0

    # Redistribute weights
    total_weight = sum(base_weights[k] for k in active)
    final_score = sum(
        active[k] * (base_weights[k] / total_weight)
        for k in active
    )

    return float(final_score)


def verdict_from_score(score: float) -> tuple:
    if score >= 60:
        return "REAL", "✅"
    elif score >= 40:
        return "SUSPICIOUS", "⚠️"
    else:
        return "DEEPFAKE", "❌"


def _score_bar(score: float, width: int = 30) -> str:
    filled = int(round(score / 100 * width))
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.1f}%"


def _score_label(score: float) -> str:
    if score >= 70:
        return "🟢 REAL"
    elif score >= 45:
        return "🟡 SUSPICIOUS"
    else:
        return "🔴 FAKE"


# ─────────────────────────────────────────────────
#  Main Entry Point
# ─────────────────────────────────────────────────

def detect(video_path: str):
    add_ffmpeg_to_path()

    video_path = os.path.abspath(video_path)

    _banner("🔍 TruthLens AI — Combined Deepfake Detector v2.0")

    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    print(f"📹 Video : {os.path.basename(video_path)}")
    print(f"📂 Path  : {video_path}")
    print(f"☁️  Mode  : {'Cloud (3-signal)' if IS_CLOUD else 'Local (4-signal)'}")

    # Run all analysers
    syncnet_score = get_syncnet_score(video_path)
    texture_score = get_texture_score(video_path)
    blink_score   = get_blink_score(video_path)
    lip_score     = get_lip_score(video_path)

    # Combine
    final_score   = combine_scores(syncnet_score, texture_score,
                                   blink_score, lip_score)
    verdict, icon = verdict_from_score(final_score)

    # Results Dashboard
    _banner("📊 FINAL RESULTS DASHBOARD")

    print(f"\n  {'Signal':<28}{'Score':>8}   {'Bar':>35}")
    print(f"  {'─'*75}")

    def _row(name, score):
        if score is None:
            print(f"  {name:<28}{'N/A':>8}   [{'░'*20}]  ⚫ Disabled")
        else:
            bar   = _score_bar(score, 20)
            label = _score_label(score)
            print(f"  {name:<28}{score:>6.1f}/100   {bar}   {label}")

    _row("SyncNet Lip-Sync",   syncnet_score)
    _row("Face Texture",       texture_score)
    _row("Blink Pattern",      blink_score)
    _row("Lip Reader",         lip_score)

    print(f"\n  {'─'*75}")
    bar = _score_bar(final_score, 20)
    print(f"\n  {'COMBINED SCORE':<28}{final_score:>6.1f}/100   {bar}")

    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  {icon}  VERDICT : {verdict:<26}     ║")
    print(f"  ╚══════════════════════════════════════╝")

    print("\n  📝 Interpretation:")
    if verdict == "REAL":
        print("     All signals indicate this is a genuine video.")
    elif verdict == "SUSPICIOUS":
        print("     Some signals are inconsistent — manual review recommended.")
    else:
        print("     Multiple signals flag this as likely AI-generated / manipulated.")

    print("\n  ✅ TruthLens AI Analysis Complete!")
    print("═" * 56 + "\n")

    return {
        'syncnet_score': syncnet_score if syncnet_score is not None else 50.0,
        'texture_score': texture_score if texture_score is not None else 50.0,
        'blink_score':   blink_score   if blink_score   is not None else 50.0,
        'lip_score':     lip_score     if lip_score     is not None else 50.0,
        'final_score':   final_score,
        'verdict':       verdict,
    }


# ─────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        vpath = sys.argv[1]
    else:
        vpath = os.path.join(BASE_DIR, "test_video.mp4")

    detect(vpath)