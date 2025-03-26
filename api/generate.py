import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import List, Dict, Any, Optional
import logging
from huggingface_hub import HfApi
import gc
from pathlib import Path
import hashlib
import re
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get HF token from environment
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise EnvironmentError("HF_TOKEN environment variable is not set")

# Initialize Hugging Face API
api = HfApi(token=hf_token)

# Model configuration
MODEL_NAME = "animaratio/arpochat"
MODEL_PATH = "model.pt"
MODEL_INFO = {
    "name": "ArpoChat",
    "base_model": "gpt2",
    "vocab_size": 50257,  # Same as GPT-2
    "total_params": 124439808,  # Same as GPT-2 since it's fine-tuned
    "layers": 12,
    "max_context": 1024,
    "embedding_size": 768,
    "training_data": "Argentine poetry of last 30 years including visual artists",
    "model_size": "497.8 MB",  # From HF repo
    "architecture": "GPT-2 Base fine-tuned",
    "license": "artistic-2.0"
}

CACHE_DIR = Path("models")
CACHE_DIR.mkdir(exist_ok=True)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Initialize tokenizer and model
_tokenizer = None
_model = None

def get_model_hash():
    """Generate a hash of the model file to check for changes."""
    try:
        model_path = CACHE_DIR / MODEL_PATH
        if model_path.exists():
            with open(model_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.warning(f"Error getting model hash: {str(e)}")
    return None

def load_model():
    """Initialize the model and tokenizer if not already loaded."""
    global _tokenizer, _model
    
    if _tokenizer is None or _model is None:
        try:
            logger.info("Starting model initialization...")
            logger.info(f"Using device: {device}")
            
            # Initialize tokenizer and model with GPT-2 base
            logger.info("Loading tokenizer from GPT-2...")
            _tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            
            # Set up padding token
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
                logger.info("Set padding token to EOS token")
            
            logger.info("Loading base model from GPT-2...")
            _model = GPT2LMHeadModel.from_pretrained("gpt2")
            
            # Check if model is already cached
            model_path = CACHE_DIR / MODEL_PATH
            if model_path.exists():
                logger.info("Found cached model, loading from cache...")
            else:
                logger.info(f"Downloading model from {MODEL_NAME}...")
                model_path = api.hf_hub_download(
                    repo_id=MODEL_NAME,
                    filename=MODEL_PATH,
                    local_dir=CACHE_DIR,
                    local_dir_use_symlinks=False
                )
                logger.info(f"Model downloaded to: {model_path}")
            
            # Load the state dictionary
            logger.info("Loading state dictionary...")
            state_dict = torch.load(model_path, map_location=device)
            logger.info("Applying state dictionary to model...")
            _model.load_state_dict(state_dict)
            
            # Set model to evaluation mode
            logger.info("Setting model to evaluation mode...")
            _model.eval()
            
            # Move model to appropriate device
            logger.info(f"Moving model to {device}...")
            _model = _model.to(device)
            
            logger.info("Model and tokenizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

# Load model on module import
load_model()

def get_model():
    """Initialize or return the model."""
    global _model, _tokenizer
    
    try:
        if _model is None:
            logger.info("Initializing model and tokenizer...")
            
            # Model path (adjust if needed)
            model_path = "model"
            if not os.path.exists(model_path):
                raise ValueError(f"Model path not found: {model_path}")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            _tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Set padding token if not set
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            
            # Load model
            logger.info("Loading model...")
            _model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Move model to device
            _model = _model.to(device)
            logger.info(f"Model loaded and moved to device: {device}")
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return _model
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def get_model_info():
    """Get information about the model."""
    model = get_model()
    return {
        "name": "ArPoChat",
        "device": str(device),
        "vocab_size": len(_tokenizer) if _tokenizer else None,
        "model_size": sum(p.numel() for p in model.parameters()),
        "requires_grad": any(p.requires_grad for p in model.parameters())
    }

def generate_text(
    prompt,
    temperatures=[0.1, 0.5, 0.9],
    max_length=512,
    **kwargs
):
    """Generate text based on the prompt with multiple temperatures."""
    try:
        results = []
        
        # Tokenize input once
        inputs = _tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=128  # Limit input length
        ).to(_model.device)
        
        # Generate for each temperature
        for temp in temperatures:
            outputs = _model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=384,  # Fixed output length
                temperature=temp,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=_tokenizer.pad_token_id,
                eos_token_id=_tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                length_penalty=1.5
            )
            
            # Decode and clean up
            generated_text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text[len(prompt):].strip()
            
            results.append({
                "text": generated_text,
                "temperature": temp
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise 