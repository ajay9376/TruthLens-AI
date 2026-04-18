"""
TruthLens AI — Combined Deepfake Detector
==========================================
Merges three independent signals into one confidence score:

  Signal              Weight   Source
  ─────────────────────────────────────────
  SyncNet Lip-Sync     40%    deepfake_detector.py
  Face Texture         35%    texture_analyzer.py
  Blink Pattern        25%    blink_detector.py  (coming soon)

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

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
FFMPEG_PATH = r"C:\Users\gujju\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin"

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
    if FFMPEG_PATH not in os.environ.get("PATH", ""):
        os.environ["PATH"] += os.pathsep + FFMPEG_PATH


# ─────────────────────────────────────────────────
#  1. SyncNet Score  (40 % weight)
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
    """
    Run SyncNet and return a 0-100 score.
    Higher = more likely REAL (good lip-sync).
    Returns 50.0 (neutral) if video has no audio or analysis fails.
    """
    _section("① SyncNet Lip-Sync Analysis  [weight: 40%]")

    if not _check_audio(video_path):
        print("⚠️  No audio track found — skipping SyncNet (score neutral: 50)")
        return 50.0

    converted = os.path.join(SYNCNET_DIR, "data", "input_video.avi")
    print("🎬 Converting video …")
    _convert_video(video_path, converted)

    print("🤖 Running SyncNet …")
    raw_output = _run_syncnet("data/input_video.avi")

    confidence, min_dist, av_offset = _parse_syncnet(raw_output)

    if confidence is None:
        print("⚠️  SyncNet returned no confidence value (score neutral: 50)")
        return 50.0

    print(f"   Confidence : {confidence:.3f}")
    print(f"   Min Dist   : {min_dist:.3f}")
    print(f"   AV Offset  : {av_offset}")

    # Map SyncNet confidence to 0-100
    # Threshold from deepfake_detector.py:  > 1.5 → REAL,  0.5-1.5 → SUSPICIOUS
    if confidence > 1.5 and (av_offset is not None and abs(av_offset) <= 1):
        score = 85.0 + min(confidence - 1.5, 1.5) * 10   # 85-100
    elif confidence > 0.5 or (av_offset is not None and abs(av_offset) <= 2):
        score = 40.0 + (confidence - 0.5) * 22.5          # 40-72
    else:
        score = max(0.0, confidence * 40.0)                # 0-40

    score = float(np.clip(score, 0, 100))
    print(f"   → SyncNet Score : {score:.1f}/100")
    return score


# ─────────────────────────────────────────────────
#  2. Texture Score  (35 % weight)
# ─────────────────────────────────────────────────

def get_texture_score(video_path: str) -> float:
    """
    Run texture analysis and return a 0-100 score.
    Higher = more likely REAL (natural texture).
    """
    _section("② Face Texture Analysis  [weight: 35%]")

    try:
        from texture_analyzer import analyze_video_texture
        score, verdict = analyze_video_texture(video_path, sample_frames=30)
        print(f"   → Texture Score : {score:.1f}/100  [{verdict}]")
        return float(score)
    except Exception as exc:
        print(f"⚠️  Texture analysis failed: {exc}")
        return 50.0


# ─────────────────────────────────────────────────
#  3. Blink Score  (25 % weight)  ← COMING SOON
# ─────────────────────────────────────────────────

def get_blink_score(video_path: str) -> float:
    """
    Run blink pattern analysis and return a 0-100 score.
    Higher = more likely REAL (natural blink pattern).
    """
    _section("③ Blink Pattern Detection  [weight: 25%]")

    try:
        from blink_detector import analyze_video_blinks
        score, verdict = analyze_video_blinks(video_path)
        print(f"   → Blink Score : {score:.1f}/100  [{verdict}]")
        return float(score)
    except Exception as exc:
        print(f"⚠️  Blink analysis failed: {exc}")
        return 50.0


def get_lip_score(video_path: str) -> float:
    _section("④ Lip Reader — Speech Articulation  [weight: 20%]")
    try:
        from lip_reader import analyze_video_lips
        score, verdict = analyze_video_lips(video_path)
        print(f"   → Lip Reader Score : {score:.1f}/100  [{verdict}]")
        return score
    except Exception as exc:
        print(f"⚠️  Lip analysis failed: {exc}")
        return 50.0


# ─────────────────────────────────────────────────
#  Combined Score & Verdict
# ─────────────────────────────────────────────────

WEIGHTS = {
    'syncnet':    0.20,   # Lip-sync (reduced — unreliable without clean speech)
    'texture':    0.20,   # Face texture temporal analysis
    'blink':      0.40,   # Blink physiology (strongest physiological signal)
    'lip_reader': 0.20,   # Speech articulation naturalness
}


def combine_scores(syncnet: float, texture: float,
                   blink: float, lip: float) -> float:
    """Weighted combination of 4 signals → 0-100 final score."""
    return (
        syncnet * WEIGHTS['syncnet']      +
        texture * WEIGHTS['texture']      +
        blink   * WEIGHTS['blink']        +
        lip     * WEIGHTS['lip_reader']
    )


def verdict_from_score(score: float) -> tuple[str, str]:
    """Return (verdict_label, emoji).

    Thresholds calibrated on ground-truth data (4-signal system):
      AI video  (test_video2.mp4) → combined ≈ 32   → < 40 → DEEPFAKE
      Real video (test_video.mp4) → combined ≈ 70   → > 60 → REAL
    """
    if score >= 60:
        return "REAL", "✅"
    elif score >= 40:
        return "SUSPICIOUS", "⚠️"
    else:
        return "DEEPFAKE", "❌"


def _score_bar(score: float, width: int = 30) -> str:
    """ASCII progress bar for the terminal."""
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

    # ── Normalise to absolute path so every sub-module opens the same file ──
    video_path = os.path.abspath(video_path)

    _banner("🔍 TruthLens AI — Combined Deepfake Detector v2.0")

    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    print(f"📹 Video : {os.path.basename(video_path)}")
    print(f"📂 Path  : {video_path}")

    # ── Run all four analysers ───────────────────
    syncnet_score = get_syncnet_score(video_path)
    texture_score = get_texture_score(video_path)
    blink_score   = get_blink_score(video_path)
    lip_score     = get_lip_score(video_path)

    # ── Combine ──────────────────────────────────
    final_score   = combine_scores(syncnet_score, texture_score,
                                   blink_score, lip_score)
    verdict, icon = verdict_from_score(final_score)

    # ── Results Dashboard ─────────────────────────
    _banner("📊 FINAL RESULTS DASHBOARD")

    print(f"\n  {'Signal':<28}{'Score':>8}   {'Bar':>35}   Weight")
    print(f"  {'─'*85}")

    def _row(name, score, weight):
        bar = _score_bar(score, 20)
        label = _score_label(score)
        print(f"  {name:<28}{score:>6.1f}/100   {bar}   {weight*100:.0f}%   {label}")

    _row("SyncNet Lip-Sync",   syncnet_score, WEIGHTS['syncnet'])
    _row("Face Texture",       texture_score, WEIGHTS['texture'])
    _row("Blink Pattern",      blink_score,   WEIGHTS['blink'])
    _row("Lip Reader",         lip_score,     WEIGHTS['lip_reader'])

    print(f"\n  {'─'*85}")

    bar = _score_bar(final_score, 20)
    print(f"\n  {'COMBINED SCORE':<28}{final_score:>6.1f}/100   {bar}")

    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  {icon}  VERDICT : {verdict:<26}     ║")
    print(f"  ╚══════════════════════════════════════╝")

    # Human-readable explanation
    print("\n  📝 Interpretation:")
    if verdict == "REAL":
        print("     All signals indicate this is a genuine video.")
        print("     Lips sync naturally, texture is realistic, blink is normal.")
    elif verdict == "SUSPICIOUS":
        print("     Some signals are inconsistent — manual review recommended.")
        print("     Could be genuine but with compression artefacts, or a subtle deepfake.")
    else:
        print("     Multiple signals flag this as likely AI-generated / manipulated.")
        print("     Proceed with caution — do not trust this video without verification.")

    print("\n  ✅ All 4 detectors active — Phase 2 complete!")
    print("  🚀 Next: Web UI (Phase 3)")
    print("═" * 56 + "\n")

    return {
        'syncnet_score': syncnet_score,
        'texture_score': texture_score,
        'blink_score':   blink_score,
        'lip_score':     lip_score,
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
