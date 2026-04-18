# 🔍 TruthLens AI — Deepfake Detection System

> *"See the Truth Behind Every Video"*

![TruthLens AI](https://img.shields.io/badge/TruthLens-AI-v2.0-purple)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Overview

TruthLens AI is an advanced deepfake detection system that uses **4 powerful AI signals** to determine if a video is real or AI-generated.

## 🎯 4-Signal Detection System

| Signal | Weight | What it Detects |
|--------|--------|-----------------|
| 🎵 SyncNet Lip-Sync | 20% | Audio-visual lip sync mismatch |
| 🎨 Face Texture | 20% | Unnatural skin texture patterns |
| 👁️ Blink Pattern | 40% | Robotic/unnatural eye blinking |
| 👄 Lip Reader | 20% | Over-animated lip movements |

## ✅ Results

| Video Type | Combined Score | Verdict |
|-----------|---------------|---------|
| Real Video | 69.8/100 | ✅ REAL |
| AI Generated | 31.1/100 | ❌ DEEPFAKE |

## 🛠️ Tech Stack

- **Python 3.10**
- **OpenCV** — Video processing
- **MediaPipe** — Face & landmark detection
- **YOLOv8** — Face detection
- **SyncNet** — Lip sync analysis
- **LibROSA** — Audio processing
- **FastAPI** — Backend API
- **HTML/CSS/JS** — Frontend UI

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/ajay9376/TruthLens-AI.git
cd TruthLens-AI

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install opencv-python mediapipe numpy Pillow
pip install librosa moviepy matplotlib scipy
pip install ultralytics syncnet-python
pip install fastapi uvicorn python-multipart
```

## 🚀 Run

```bash
python api.py
```

Open browser at `http://localhost:8000`

## 📊 How It Works

Video Input
↓
4 Parallel Analysers
↓
SyncNet + Texture + Blink + Lip Reader
↓
Weighted Combined Score
↓
REAL ✅ or DEEPFAKE ❌

## 🌍 Real World Use Cases

- 🚔 Police — Verify video evidence
- 📰 Journalists — Fact check viral videos
- 🏦 Banks — Secure video KYC
- 👂 Deaf Community — Lip reading from silent videos
- 🗳️ Election — Detect political deepfakes

## 👨‍💻 Built By

**Gujjula Ajay Kumar** — AI/ML Developer

---

⭐ Star this repo if you found it useful!