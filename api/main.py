from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
import os
import uvicorn
from .generate import generate_text, get_model_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENVIRONMENT = "production" if os.getenv("RENDER") else "local"
PORT = int(os.getenv("PORT", 8004))
logger.info(f"Starting ArpoChat API in {ENVIRONMENT} mode on port {PORT}")

app = FastAPI(title="ArpoChat API", description="API for poetry generation", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 1024

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": f"Model loaded and ready ({ENVIRONMENT})"}

@app.get("/model-info")
async def get_model_information():
    try:
        info = get_model_info()
        logger.info(f"Model info requested: {info}")
        return info
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        logger.info(f"Received generation request with prompt: {request.prompt}")
        generated_texts = generate_text(request.prompt, max_length=request.max_length)
        logger.info(f"Generated texts: {[t['text'][:50] + '...' for t in generated_texts]}")
        return {"generated_texts": generated_texts}
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Verify static file paths
STATIC_DIR = "public/static"
PUBLIC_DIR = "public"
logger.info(f"Mounting static files from {STATIC_DIR} and {PUBLIC_DIR}")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)