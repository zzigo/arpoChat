import os
import logging
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from fastapi import HTTPException

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model caching
model = None
tokenizer = None

# Get Hugging Face token from environment
HF_TOKEN = os.getenv("HF_TOKEN")

def get_model():
    """Load and cache the model with error handling"""
    global model, tokenizer
    
    if model is None:
        try:
            logger.info("Loading model...")
            
            # Initialize the model and tokenizer from GPT-2
            logger.debug("Loading base GPT-2 model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                token=HF_TOKEN,
                trust_remote_code=True
            )
            logger.debug("Base model loaded")
            
            tokenizer = AutoTokenizer.from_pretrained(
                "gpt2",
                token=HF_TOKEN,
                trust_remote_code=True
            )
            logger.debug("Tokenizer loaded")
            
            # Create the pipeline
            model = pipeline('text-generation', 
                           model=base_model,
                           tokenizer=tokenizer,
                           device=-1)  # Force CPU
            logger.info("Successfully loaded model")
            return model
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    return model

def generate_text(model, prompt: str, max_length: int = 100, num_return_sequences: int = 1, temperature: float = 0.7) -> str:
    """Generate text using the ML model"""
    try:
        logger.debug(f"Generating text with prompt: {prompt[:50]}...")
        outputs = model(prompt,
                       max_length=max_length,
                       temperature=temperature,
                       num_return_sequences=num_return_sequences,
                       do_sample=True)
        
        logger.debug("Text generation successful")
        return outputs[0]['generated_text']
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise

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

def generate_text_with_model(prompt: str, max_length: int, temperature: float) -> dict:
    """Generate text using the ML model"""
    try:
        model_available = get_model()
        
        if model_available:
            try:
                generated_text = generate_text(model_available, prompt, max_length, 1, temperature)
                stats = {
                    'total_chars': len(generated_text),
                    'total_words': len(generated_text.split()),
                    'total_tokens': len(tokenizer.encode(generated_text)),
                    'temperature': temperature,
                    'max_length': max_length,
                    'model_used': True
                }
                return {
                    'generated_text': generated_text,
                    'prompt': prompt,
                    'stats': stats
                }
            except Exception as model_error:
                logger.warning(f"Model generation failed: {str(model_error)}")
                # Fall back to test response if model fails
                result = generate_fallback_response(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature
                )
                return result
        else:
            logger.warning("Model not available, using fallback response")
            # Use fallback if model isn't available
            result = generate_fallback_response(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature
            )
            return result
            
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        raise Exception(f"Error in generate endpoint: {str(e)}") 