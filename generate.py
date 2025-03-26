from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
from typing import Optional
import logging
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
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

# Constants
MODEL_ID = "animaratio/arpochat"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables for model caching
_model = None
_tokenizer = None

def get_model():
    """Load and cache the model and tokenizer."""
    global _model, _tokenizer
    
    try:
        if _model is None:
            logger.info(f"Loading model from {MODEL_ID}")
            
            # Get Hugging Face token from environment
            hf_token = os.getenv('HF_TOKEN')
            if not hf_token:
                raise ValueError("HF_TOKEN environment variable not set")
            
            # Load tokenizer and model
            _tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            _model = GPT2LMHeadModel.from_pretrained(
                MODEL_ID,
                token=hf_token,
                trust_remote_code=True
            )
            
            # Configure model
            _tokenizer.pad_token = _tokenizer.eos_token
            _model.config.pad_token_id = _model.config.eos_token_id
            _model.to(DEVICE)
            _model.eval()
            
            logger.info("Model loaded successfully")
        
        return _model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def generate_text(
    model: GPT2LMHeadModel,
    prompt: str,
    max_length: Optional[int] = 100,
    temperature: Optional[float] = 0.7
) -> str:
    """Generate text using the loaded model."""
    try:
        # Encode the prompt
        inputs = _tokenizer(prompt, return_tensors='pt', padding=True)
        input_ids = inputs['input_ids'].to(DEVICE)
        attention_mask = inputs['attention_mask'].to(DEVICE)
        
        # Set generation parameters
        gen_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_length': max_length,
            'num_return_sequences': 1,
            'do_sample': True,
            'temperature': temperature,
            'top_k': 50,
            'top_p': 0.95,
            'no_repeat_ngram_size': 3,
            'num_beams': 5,
            'pad_token_id': _tokenizer.eos_token_id,
            'early_stopping': True
        }
        
        # Generate text
        with torch.no_grad():
            output_sequences = model.generate(**gen_kwargs)
        
        # Decode and clean up the response
        generated_text = _tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        generated_text = generated_text.replace(prompt, '').strip()
        
        return generated_text
        
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise

def generate_text_with_model(prompt: str, max_length: int, temperature: float) -> dict:
    """Generate text using the ML model"""
    try:
        outputs = _model(prompt,
                       max_length=max_length,
                       temperature=temperature,
                       num_return_sequences=1,
                       do_sample=True)
        
        generated_text = outputs[0]['generated_text']
        
        # Calculate stats
        stats = {
            'total_chars': len(generated_text),
            'total_words': len(generated_text.split()),
            'total_tokens': len(_tokenizer.encode(generated_text)),
            'temperature': temperature,
            'max_length': max_length,
            'model_used': True
        }
        
        return {
            'generated_text': generated_text,
            'prompt': prompt,
            'stats': stats
        }
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise Exception(f"Error generating text: {str(e)}")

def generate_fallback_response(prompt: str, max_length: int, temperature: float) -> dict:
    """Generate a fallback response when the model is not available"""
    response_text = f"Test response for prompt: {prompt}"
    return {
        'generated_text': response_text,
        'prompt': prompt,
        'stats': {
            'total_chars': len(response_text),
            'total_words': len(response_text.split()),
            'temperature': temperature,
            'max_length': max_length,
            'model_used': False
        }
    }

@app.get("/")
@app.head("/")
async def root():
    """
    Root endpoint - health check
    Supports both GET and HEAD requests
    """
    try:
        model_status = "loaded" if get_model() else "not loaded"
        response = {
            "status": "ok",
            "message": "ArpoChat API is running",
            "model_status": model_status
        }
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Error in root endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """
    Text generation endpoint with fallback
    """
    try:
        logger.info(f"Received generation request with prompt: {request.prompt[:50]}...")
        
        # Try to ensure model is loaded
        model_available = get_model()
        
        if model_available:
            try:
                # Try to generate with model
                result = generate_text_with_model(
                    prompt=request.prompt,
                    max_length=request.max_length,
                    temperature=request.temperature
                )
                logger.info("Successfully generated text with model")
                return GenerateResponse(**result)
            except Exception as model_error:
                logger.warning(f"Model generation failed: {str(model_error)}")
                # Fall back to test response if model fails
                result = generate_fallback_response(
                    prompt=request.prompt,
                    max_length=request.max_length,
                    temperature=request.temperature
                )
                return GenerateResponse(**result)
        else:
            # Use fallback if model is not available
            logger.warning("Model not available, using fallback response")
            result = generate_fallback_response(
                prompt=request.prompt,
                max_length=request.max_length,
                temperature=request.temperature
            )
            return GenerateResponse(**result)
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 