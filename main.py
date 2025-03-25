import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
from api.generate import get_model, generate_text

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the public directory
PUBLIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public")
STATIC_DIR = os.path.join(PUBLIC_DIR, "static")

# Mount static files directory
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")

class GenerateRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 100
    temperatures: Optional[List[float]] = [0.2, 0.5, 0.8]

@app.get("/")
async def root():
    """Serve the main HTML page"""
    index_path = os.path.join(PUBLIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Index file not found")
    return FileResponse(index_path)

@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text based on the prompt with multiple temperatures"""
    try:
        logger.info(f"Received generation request with prompt: {request.prompt[:50]}...")
        
        # Get the model (this will load it if not already loaded)
        model = get_model()
        if not model:
            logger.error("Failed to load model")
            raise HTTPException(status_code=500, detail="Failed to load model")
        
        # Generate text for each temperature
        responses = []
        total_chars = 0
        total_words = 0
        
        for temp in request.temperatures:
            generated_text = generate_text(
                model=model,
                prompt=request.prompt,
                max_length=request.max_length,
                temperature=temp
            )
            
            responses.append({
                "temperature": temp,
                "text": generated_text
            })
            
            total_chars += len(generated_text)
            total_words += len(generated_text.split())
        
        logger.info("Text generation successful")
        return JSONResponse(content={
            "success": True,
            "prompt": request.prompt,
            "responses": responses,
            "stats": {
                "total_chars": total_chars,
                "total_words": total_words,
                "avg_words_per_response": total_words / len(request.temperatures)
            }
        })
        
    except Exception as e:
        logger.error(f"Error in /generate endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "Lo siento, hubo un error generando la respuesta"
            }
        )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"} 