from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
import tempfile
import os
import sys
import urllib.request
from datetime import datetime

# ─── Fix paths ───
os.environ["PATH"] += ":/usr/bin:/usr/local/bin"

# Windows ffmpeg path
FFMPEG_PATH = r"C:\Users\gujju\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin"
if os.path.exists(FFMPEG_PATH):
    os.environ["PATH"] += os.pathsep + FFMPEG_PATH

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ─── Download SyncNet model if not exists ───
model_path = "syncnet_python/data/syncnet_v2.model"
if not os.path.exists(model_path):
    print("Downloading SyncNet model...")
    os.makedirs("syncnet_python/data", exist_ok=True)
    try:
        urllib.request.urlretrieve(
            "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model",
            model_path
        )
        print("Model downloaded!")
    except Exception as e:
        print(f"Model download failed: {e}")

# ─── App ───
app = FastAPI(title="TruthLens AI")

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── Routes ───
@app.get("/")
async def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)


@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from combined_detector import detect
        results = detect(tmp_path)

        if results:
            return JSONResponse({
                "verdict":       results["verdict"],
                "final_score":   results["final_score"],
                "syncnet_score": results["syncnet_score"],
                "texture_score": results["texture_score"],
                "blink_score":   results["blink_score"],
                "lip_score":     results["lip_score"],
            })
        else:
            return JSONResponse({"error": "Analysis failed"}, status_code=500)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        os.unlink(tmp_path)


@app.post("/analyze-with-report")
async def analyze_with_report(video: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from combined_detector import detect
        from report_generator import generate_report

        results = detect(tmp_path)

        if results:
            # Generate PDF report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"/tmp/TruthLens_Report_{timestamp}.pdf"
            generate_report(results, video.filename, report_path)

            return JSONResponse({
                "verdict":       results["verdict"],
                "final_score":   results["final_score"],
                "syncnet_score": results["syncnet_score"],
                "texture_score": results["texture_score"],
                "blink_score":   results["blink_score"],
                "lip_score":     results["lip_score"],
                "report_path":   report_path,
            })
        else:
            return JSONResponse({"error": "Analysis failed"}, status_code=500)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        os.unlink(tmp_path)


@app.get("/download-report")
async def download_report(path: str):
    if os.path.exists(path):
        return FileResponse(
            path,
            media_type="application/pdf",
            filename="TruthLens_Report.pdf"
        )
    return JSONResponse({"error": "Report not found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)