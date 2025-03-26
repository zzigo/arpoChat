from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
from typing import Optional, List, Dict, Any
import logging
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from huggingface_hub import hf_hub_download

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

# Enable CORS with proper configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8004",
        "http://localhost:3000",
        "http://127.0.0.1:8004",
        "http://127.0.0.1:3000",
        "https://*.onrender.com",  # Allow Render subdomains
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Constants
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model", "model.pt")
HF_MODEL_REPO = "animaratio/arpochat"  # Your Hugging Face model repository
HF_MODEL_FILENAME = "model.pt"  # The filename in your HF repo
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables for model caching
_model = None
_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # Initialize tokenizer immediately
_tokenizer.pad_token = _tokenizer.eos_token  # Configure tokenizer

def load_model_weights():
    """Load model weights either from local path or Hugging Face"""
    # First try local path (for development)
    if os.path.exists(LOCAL_MODEL_PATH):
        logger.info(f"Loading model from local path: {LOCAL_MODEL_PATH}")
        return torch.load(LOCAL_MODEL_PATH, map_location=DEVICE)
    
    # If local file not found, try Hugging Face (for production)
    try:
        logger.info(f"Local model not found, downloading from Hugging Face: {HF_MODEL_REPO}")
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not set")
            
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=HF_MODEL_FILENAME,
            token=hf_token
        )
        return torch.load(model_path, map_location=DEVICE)
    except Exception as e:
        logger.error(f"Error downloading model from Hugging Face: {str(e)}")
        raise

def get_model():
    """Load and cache the model and tokenizer."""
    global _model
    
    try:
        if _model is None:
            # Initialize model architecture
            _model = GPT2LMHeadModel.from_pretrained('gpt2')
            
            # Load custom weights (either local or from HF)
            state_dict = load_model_weights()
            _model.load_state_dict(state_dict)
            
            # Configure model
            _model.config.pad_token_id = _model.config.eos_token_id
            _model.to(DEVICE)
            _model.eval()
            
            logger.info("Model loaded successfully")
        
        return _model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

class GenerateRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 1000
    temperatures: Optional[List[float]] = [0.7]

class GenerateResponse(BaseModel):
    success: bool
    prompt: str
    responses: List[Dict[str, Any]]
    stats: Optional[Dict[str, Any]] = None

def generate_text(
    model: GPT2LMHeadModel,
    prompt: str,
    max_length: Optional[int] = 1000,
    temperature: Optional[float] = 0.7
) -> str:
    """Generate text using the loaded model with enhanced semantic coherence."""
    try:
        # Analyze prompt for key elements
        prompt_tokens = _tokenizer.encode(prompt, add_special_tokens=False)
        prompt_length = len(prompt_tokens)
        
        # Encode the prompt with special handling
        inputs = _tokenizer(prompt, return_tensors='pt', padding=True)
        input_ids = inputs['input_ids'].to(DEVICE)
        attention_mask = inputs['attention_mask'].to(DEVICE)
        
        # Enhanced generation parameters for semantic bridging
        gen_kwargs = {
            'input_ids': input_ids,
            'max_length': max_length + prompt_length,  # Account for prompt length
            'num_return_sequences': 1,
            'do_sample': True,
            'temperature': temperature,
            'top_k': 40,  # More focused word choice for coherence
            'top_p': 0.85,  # More selective sampling
            'no_repeat_ngram_size': 3,  # Allow some repetition for style
            'num_beams': 3,  # Reduced for more natural flow
            'length_penalty': 1.3,  # Encourage development of ideas
            'repetition_penalty': 1.15,  # Slightly increased to avoid exact prompt repetition
            'pad_token_id': _tokenizer.eos_token_id,
            'eos_token_id': _tokenizer.eos_token_id,
            'early_stopping': True
        }
        
        # Generate initial text
        with torch.no_grad():
            output_sequences = model.generate(**gen_kwargs)
        
        # Process the generated text
        generated_text = _tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        
        # Remove the prompt but maintain flow
        if generated_text.startswith(prompt):
            # Find a good breakpoint after the prompt
            prompt_end = len(prompt)
            next_sentence = generated_text.find('. ', prompt_end)
            next_line = generated_text.find('\n', prompt_end)
            
            # Choose the closer breakpoint
            if next_sentence != -1 and next_line != -1:
                break_point = min(next_sentence, next_line)
            else:
                break_point = max(next_sentence, next_line)
                
            if break_point == -1:
                break_point = prompt_end
            
            # Remove the prompt and any partial sentence/line
            generated_text = generated_text[break_point + 1:].strip()
        
        # If the text is too short or lacks coherence
        if len(generated_text.split()) < max_length // 8:
            gen_kwargs.update({
                'temperature': min(temperature + 0.1, 0.95),
                'top_p': 0.9,
                'repetition_penalty': 1.1
            })
            with torch.no_grad():
                output_sequences = model.generate(**gen_kwargs)
            generated_text = _tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
        
        return generated_text.strip()
        
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise

def generate_fallback_response(prompt: str, max_length: int, temperature: float) -> dict:
    """Generate a fallback response when the model is not available"""
    response_text = f"Test response for prompt: {prompt}"
    return {
        'success': True,
        'prompt': prompt,
        'responses': [{
            'temperature': temperature,
            'text': response_text
        }],
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

@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Text generation endpoint with fallback
    """
    try:
        logger.info(f"Received generation request with prompt: {request.prompt[:50]}...")
        
        # Validate input
        if not request.prompt.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Empty prompt provided"
                }
            )
        
        # Try to ensure model is loaded
        model = get_model()
        if not model:
            logger.error("Model failed to load")
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "error": "Model not available"
                }
            )
        
        try:
            # Generate responses for each temperature
            responses = []
            total_chars = 0
            total_words = 0
            total_tokens = 0
            
            for temp in request.temperatures:
                try:
                    generated_text = generate_text(
                        model=model,
                        prompt=request.prompt,
                        max_length=request.max_length,
                        temperature=temp
                    )
                    
                    # Calculate stats
                    chars = len(generated_text)
                    words = len(generated_text.split())
                    tokens = len(_tokenizer.encode(generated_text))
                    
                    total_chars += chars
                    total_words += words
                    total_tokens += tokens
                    
                    responses.append({
                        'temperature': temp,
                        'text': generated_text,
                        'stats': {
                            'chars': chars,
                            'words': words,
                            'tokens': tokens
                        }
                    })
                except Exception as gen_error:
                    logger.error(f"Error generating text for temperature {temp}: {str(gen_error)}")
                    responses.append({
                        'temperature': temp,
                        'text': f"Error generating text: {str(gen_error)}",
                        'error': True
                    })
            
            # If we have at least one successful generation
            if any(not response.get('error', False) for response in responses):
                return GenerateResponse(
                    success=True,
                    prompt=request.prompt,
                    responses=responses,
                    stats={
                        'total_chars': total_chars,
                        'total_words': total_words,
                        'total_tokens': total_tokens,
                        'model_used': True
                    }
                )
            else:
                # If all generations failed, return fallback
                return generate_fallback_response(
                    prompt=request.prompt,
                    max_length=request.max_length,
                    temperature=request.temperatures[0] if request.temperatures else 0.7
                )
                
        except Exception as gen_error:
            logger.error(f"Error during generation process: {str(gen_error)}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": str(gen_error),
                    "message": "Error during text generation"
                }
            )
            
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "Lo siento, hubo un error generando la respuesta"
            }
        ) 