from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from typing import Optional
import logging
from .generate import generate_text, get_model

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

# Mount static files
app.mount("/static", StaticFiles(directory="public/static"), name="static")
app.mount("/", StaticFiles(directory="public", html=True), name="public")

class GenerateRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.8

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

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """Generate text from a prompt."""
    try:
        logger.info(f"Received generation request with prompt: {request.prompt[:50]}...")
        
        # Validate input
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        # Generate text
        responses = generate_text(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        return JSONResponse(
            content={
                "status": "success",
                "responses": responses,
                "prompt": request.prompt
            }
        )
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error generating text: {str(e)}"
            }
        ) 