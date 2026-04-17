import os
import io
import uuid
import torch
import yt_dlp
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.inference.predictor import ChordPredictor
from src.utils.config_loader import load_config, Config

def create_app():
    # 1. Load configuration
    config_dict = load_config()
    cfg = Config(config_dict)
    
    app = FastAPI(title="Deep Chord Project API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], 
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # System directories
    MODELS_DIR = cfg.training.model_dir
    TEMP_DIR = "temp_audio"
    WEB_APP_DIR = "web_app"
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")
    
    # Mount the frontend directory
    app.mount("/frontend", StaticFiles(directory=WEB_APP_DIR), name="frontend")

    # --- GLOBAL PREDICTOR STATE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = ChordPredictor(device=device, config=cfg)
    current_model_name = ""

    def auto_load_best_model():
        nonlocal current_model_name
        files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")]
        if not files:
            return

        # Prioritize "best_chord_model.pth" if it exists
        if "best_chord_model.pth" in files:
            best_model = "best_chord_model.pth"
        else:
            # Sort all non-multitask "best" models (usually has score in name)
            best_candidates = sorted([f for f in files if "best" in f and "multitask" not in f])
            if best_candidates:
                best_model = best_candidates[0]
            else:
                # Try enhanced
                enhanced_candidates = sorted([f for f in files if "enhanced" in f])
                if enhanced_candidates:
                    best_model = enhanced_candidates[0]
                else:
                    # Fallback to anything
                    best_model = sorted(files)[0]

        try:
            predictor.load_model(os.path.join(MODELS_DIR, best_model))
            current_model_name = best_model
            print(f"✅ Auto-loaded best model: {best_model}")
        except Exception as e:
            print(f"❌ Failed to auto-load {best_model}: {e}")

    auto_load_best_model()

    # --- ROUTES ---

    @app.get("/")
    async def root():
        """Redirect or serve the main index.html"""
        from fastapi.responses import FileResponse
        return FileResponse(os.path.join(WEB_APP_DIR, "index.html"))

    @app.get("/list-models")
    async def list_models():
        files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")]
        return {"models": files, "current": current_model_name}

    @app.post("/load-model/{model_name}")
    async def load_model_endpoint(model_name: str):
        nonlocal current_model_name
        path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Model file not found")
        try:
            predictor.load_model(path)
            current_model_name = model_name
            return {"status": "success", "message": f"Model {model_name} loaded"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    class SearchRequest(BaseModel):
        query: str

    @app.post("/search-youtube")
    async def search_youtube(request: SearchRequest):
        try:
            ydl_opts = {'extract_flat': True, 'quiet': True, 'no_warnings': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"ytsearch5:{request.query}", download=False)
                results = []
                for entry in info.get('entries', []):
                    thumbnails = entry.get('thumbnails', [])
                    results.append({
                        "id": entry.get('id'),
                        "title": entry.get('title'),
                        "url": entry.get('url'),
                        "thumbnail": thumbnails[-1]['url'] if thumbnails else ""
                    })
            return {"status": "success", "results": results}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @app.post("/analyze")
    async def analyze_audio(file: UploadFile = File(...)):
        try:
            temp_path = f"{TEMP_DIR}/{uuid.uuid4()}.wav"
            content = await file.read()
            with open(temp_path, "wb") as f:
                f.write(content)
            
            chord_names = predictor.predict_audio(temp_path)
            regions = predictor.format_to_regions(chord_names)
            
            os.remove(temp_path) # Clean up
            return {"status": "success", "chords": regions}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    class YouTubeRequest(BaseModel):
        url: str

    @app.post("/analyze-youtube")
    async def analyze_youtube(request: YouTubeRequest):
        try:
            file_id = str(uuid.uuid4())
            wav_path_no_ext = f"{TEMP_DIR}/{file_id}"
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': wav_path_no_ext + ".%(ext)s",
                'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
                'quiet': True, 'no_warnings': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([request.url])

            actual_wav_path = wav_path_no_ext + ".wav"
            chord_names = predictor.predict_audio(actual_wav_path)
            regions = predictor.format_to_regions(chord_names)
            
            return {
                "status": "success", 
                "chords": regions,
                "audio_url": f"http://localhost:8000/temp/{file_id}.wav"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return app
