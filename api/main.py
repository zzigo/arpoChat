from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from typing import Optional, List
import logging
from .generate import generate_text, get_model, get_model_info, _tokenizer, _model
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ArpoChat API",
    description="API for ArpoChat text generation",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.8
    custom_prompt: Optional[str] = None

class GenerationRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    max_length: Optional[int] = 3000
    top_k: Optional[int] = 100
    top_p: Optional[float] = 0.9
    repetition_penalty: Optional[float] = 1.5
    custom_prompt: Optional[str] = None

@app.get("/health")
async def health_check():
    """Health check endpoint to verify model status."""
    try:
        model = get_model()
        if model is None:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "message": "Model not loaded"}
            )
        return JSONResponse(
            content={"status": "ok", "message": "Model loaded and ready"}
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": str(e)}
        )

@app.get("/model-info")
async def get_model_information():
    """Get information about the loaded model."""
    try:
        return get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate(request: Request):
    """Generate text based on the prompt."""
    try:
        # Get request body
        body = await request.json()
        prompt = body.get("prompt", "")
        
        # Log the request
        logger.info(f"Received generation request with prompt: {prompt}")
        
        # Extract generation parameters with defaults
        params = {
            "temperature": body.get("temperature", 0.9),
            "max_length": body.get("max_length", 1024),
            "num_beams": body.get("num_beams", 4),
            "no_repeat_ngram_size": body.get("no_repeat_ngram_size", 3),
            "length_penalty": body.get("length_penalty", 2.0),
            "top_k": body.get("top_k", 50),
            "top_p": body.get("top_p", 0.95),
            "repetition_penalty": body.get("repetition_penalty", 1.2)
        }
        
        # Generate text
        generated_text = generate_text(prompt, **params)
        
        # Return response
        return {
            "generated_texts": [generated_text],
            "model_info": {
                "name": "ArPoChat",
                "vocab_size": len(_tokenizer),
                "total_params": sum(p.numel() for p in _model.parameters()),
                "n_layer": _model.config.n_layer,
                "device": str(_model.device),
                "n_positions": _model.config.n_positions,
                "n_embd": _model.config.n_embd
            }
        }
        
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating text: {str(e)}"
        )

# Mount static files AFTER all API routes
app.mount("/static", StaticFiles(directory="public/static"), name="static")
app.mount("/", StaticFiles(directory="public", html=True), name="public") 