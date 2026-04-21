"""
TruthLens AI — AI Image Detector v1.0
======================================
Detects if an image is AI generated using:

  ① Noise Analysis      — AI images have unnatural noise
  ② Frequency Analysis  — AI images have GAN frequency patterns
  ③ Texture Analysis    — AI images are too perfect
  ④ Edge Analysis       — AI images have unnatural edges
  ⑤ Color Analysis      — AI images have unusual color distribution
"""

import cv2
import numpy as np
import os
import sys
import io
from scipy import signal
from scipy.fft import fft2, fftshift

# Fix emoji output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_image(image_path: str):
    """Load and preprocess image"""
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return rgb, gray


def analyze_noise(gray: np.ndarray) -> dict:
    """Analyze noise patterns"""
    # Real images have natural sensor noise
    # AI images have unusual noise patterns

    # High frequency noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray.astype(float) - blurred.astype(float)

    noise_std  = float(np.std(noise))
    noise_mean = float(np.mean(np.abs(noise)))
    noise_kurt = float(np.mean((noise - noise.mean())**4) / (noise.std()**4 + 1e-6))

    return {
        'noise_std':  noise_std,
        'noise_mean': noise_mean,
        'noise_kurt': noise_kurt,
    }


def analyze_frequency(gray: np.ndarray) -> dict:
    """Analyze frequency domain for GAN artifacts"""
    # GAN generated images have specific frequency patterns
    # especially at regular intervals (grid artifacts)

    # FFT analysis
    f = fft2(gray.astype(float))
    fshift = fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)

    # Check for periodic patterns (GAN grid artifacts)
    center_y, center_x = magnitude.shape[0]//2, magnitude.shape[1]//2

    # High frequency energy ratio
    total_energy  = float(np.sum(magnitude))
    center_energy = float(np.sum(magnitude[
        center_y-10:center_y+10,
        center_x-10:center_x+10
    ]))
    hf_ratio = 1 - (center_energy / (total_energy + 1e-6))

    freq_std  = float(np.std(magnitude))
    freq_mean = float(np.mean(magnitude))

    return {
        'hf_ratio':   hf_ratio,
        'freq_std':   freq_std,
        'freq_mean':  freq_mean,
    }


def analyze_texture(gray: np.ndarray) -> dict:
    """Analyze texture patterns"""
    # Laplacian variance — real images have natural texture
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Local Binary Pattern entropy
    resized = cv2.resize(gray, (256, 256))

    # Sobel edges
    sobelx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    edge_std  = float(np.std(edge_magnitude))
    edge_mean = float(np.mean(edge_magnitude))

    return {
        'laplacian_var': laplacian_var,
        'edge_std':      edge_std,
        'edge_mean':     edge_mean,
    }


def analyze_color(rgb: np.ndarray) -> dict:
    """Analyze color distribution"""
    # AI images often have unusual color distributions

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

    # Color channel statistics
    r_std = float(np.std(r))
    g_std = float(np.std(g))
    b_std = float(np.std(b))

    # Color correlation between channels
    rg_corr = float(np.corrcoef(r.flatten(), g.flatten())[0,1])
    rb_corr = float(np.corrcoef(r.flatten(), b.flatten())[0,1])
    gb_corr = float(np.corrcoef(g.flatten(), b.flatten())[0,1])

    avg_corr = (rg_corr + rb_corr + gb_corr) / 3

    return {
        'r_std':    r_std,
        'g_std':    g_std,
        'b_std':    b_std,
        'avg_corr': avg_corr,
    }


def score_image(noise: dict, freq: dict,
                texture: dict, color: dict) -> float:
    """Score 0-100 (higher = more REAL)"""
    score = 0

    print(f"\n📊 Image Analysis Metrics:")
    print(f"   Noise Std       : {noise['noise_std']:.3f}")
    print(f"   Noise Kurt      : {noise['noise_kurt']:.3f}")
    print(f"   HF Ratio        : {freq['hf_ratio']:.3f}")
    print(f"   Freq Std        : {freq['freq_std']:.3f}")
    print(f"   Laplacian Var   : {texture['laplacian_var']:.2f}")
    print(f"   Edge Std        : {texture['edge_std']:.2f}")
    print(f"   Color Corr      : {color['avg_corr']:.3f}")

    # ① Noise Analysis (max 30 pts)
    # AI images: noise_std often 5-8 (too clean)
    # Real images: noise_std 8-20 (natural noise)
    if noise['noise_std'] >= 12.0:
        pts = 30
    elif noise['noise_std'] >= 8.0:
        pts = 18
    elif noise['noise_std'] >= 5.0:
        pts = 8
    else:
        pts = 0
    score += pts
    print(f"\n   ① Noise Std {noise['noise_std']:.3f} → +{pts}/30 pts")

    # ② HF Ratio (max 30 pts) ← increased weight!
    if 0.85 <= freq['hf_ratio'] <= 0.97:
        pts = 30
    elif 0.80 <= freq['hf_ratio'] <= 0.98:
        pts = 15
    else:
        pts = 0   # AI image = 0 pts ✅
    score += pts
    print(f"   ② HF Ratio {freq['hf_ratio']:.3f} → +{pts}/30 pts")

    # ③ Texture Analysis (max 20 pts) ← reduced weight
    if 50 <= texture['laplacian_var'] <= 400:
        pts = 20
    elif texture['laplacian_var'] > 400:
        pts = 10  # Too sharp = possibly AI
    else:
        pts = 0
    score += pts
    print(f"   ③ Laplacian {texture['laplacian_var']:.2f} → +{pts}/20 pts")

    # ④ Color Analysis (max 20 pts)
    if 0.7 <= color['avg_corr'] <= 0.95:
        pts = 20
    elif 0.5 <= color['avg_corr'] <= 0.97:
        pts = 10
    else:
        pts = 0
    score += pts
    print(f"   ④ Color Corr {color['avg_corr']:.3f} → +{pts}/20 pts")

    total = min(int(score), 100)
    print(f"\n   TOTAL: {total}/100")
    return float(total)


def detect_ai_image(image_path: str):
    """
    Main function — detect if image is AI generated
    Returns (score, verdict)
    """
    print("\n" + "="*50)
    print("🖼️  TruthLens AI — AI Image Detector v1.0")
    print("="*50)

    # Load image
    print(f"📂 Loading: {os.path.basename(image_path)}")
    rgb, gray = load_image(image_path)

    if rgb is None:
        print("❌ Could not load image!")
        return 50.0, "UNKNOWN"

    h, w = gray.shape
    print(f"📐 Size: {w}x{h}")

    # Resize for consistent analysis
    gray_resized = cv2.resize(gray, (512, 512))
    rgb_resized  = cv2.resize(rgb,  (512, 512))

    print("🔍 Analyzing image...")

    # Run all analyses
    noise   = analyze_noise(gray_resized)
    freq    = analyze_frequency(gray_resized)
    texture = analyze_texture(gray_resized)
    color   = analyze_color(rgb_resized)

    # Score
    score = score_image(noise, freq, texture, color)

    print(f"\n🎯 AI Image Score: {score:.1f}/100")

    if score >= 65:
        print("✅ Image: REAL — Natural photograph!")
        verdict = "REAL"
    elif score >= 40:
        print("⚠️ Image: SUSPICIOUS — Possible AI generation!")
        verdict = "SUSPICIOUS"
    else:
        print("❌ Image: AI GENERATED — Synthetic image detected!")
        verdict = "FAKE"

    print("="*50)
    return score, verdict


# ─── CLI ───
if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Test with default image
        path = os.path.join(BASE_DIR, "image_test.jpg")

    if not os.path.exists(path):
        print(f"❌ Image not found: {path}")
    else:
        score, verdict = detect_ai_image(path)
        print(f"\n🏁 Final: {verdict} ({score:.1f}/100)")