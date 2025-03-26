from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
from .generate import generate_text, get_model_info
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    return {"status": "ok", "message": "Model loaded and ready"}

@app.get("/model-info")
async def get_model_information():
    try:
        return get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        logger.info(f"Received generation request with prompt: {request.prompt}")
        generated_texts = generate_text(request.prompt, max_length=request.max_length)
        return {"generated_texts": generated_texts}
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/static", StaticFiles(directory="public/static"), name="static")
app.mount("/", StaticFiles(directory="public", html=True), name="public")