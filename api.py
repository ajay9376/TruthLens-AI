from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse
import tempfile
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Add ffmpeg to path
FFMPEG_PATH = r"C:\Users\gujju\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin"
os.environ["PATH"] += os.pathsep + FFMPEG_PATH

app = FastAPI(title="TruthLens AI")

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

from starlette.responses import HTMLResponse

@app.get("/")
async def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)

@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    # Save uploaded video temporarily
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)