import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import List, Dict, Any, Optional
import logging
from huggingface_hub import HfApi
import gc
from pathlib import Path

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
CACHE_DIR = os.getenv('HF_HOME', '/tmp/.cache/huggingface')
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, 'models')

# Ensure cache directory exists
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def load_model():
    """Load the model with proper caching."""
    try:
        logger.info("Initializing tokenizer and model...")
        # First verify token is valid
        api.whoami()
        
        # Initialize with authentication
        _tokenizer = GPT2Tokenizer.from_pretrained(
            'gpt2',  # Use GPT2 tokenizer as base
            token=hf_token,
            local_files_only=False,
            cache_dir=CACHE_DIR
        )
        
        # Create a new model instance
        _model = GPT2LMHeadModel.from_pretrained(
            'gpt2',
            token=hf_token,
            local_files_only=False,
            cache_dir=CACHE_DIR
        )
        
        # Check if model is already cached
        cached_model_path = os.path.join(MODEL_CACHE_DIR, MODEL_PATH)
        if not os.path.exists(cached_model_path):
            logger.info(f"Downloading trained model from {MODEL_NAME}...")
            model_path = api.hf_hub_download(
                repo_id=MODEL_NAME,
                filename=MODEL_PATH,
                token=hf_token,
                local_dir=MODEL_CACHE_DIR,
                local_dir_use_symlinks=False,
                force_download=False
            )
        else:
            logger.info("Using cached model...")
            model_path = cached_model_path
        
        # Load the state dictionary into the model
        state_dict = torch.load(model_path, map_location='cpu')
        _model.load_state_dict(state_dict)
        _model.eval()  # Set to evaluation mode
        
        # Move model to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(device)
        
        logger.info("Model and tokenizer initialized successfully")
        return _model, _tokenizer
    except Exception as e:
        logger.error(f"Error initializing model or tokenizer: {str(e)}")
        raise

# Initialize model and tokenizer
try:
    _model, _tokenizer = load_model()
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    raise

def get_model():
    """Return the loaded model."""
    return _model

def generate_text(
    prompt: str,
    max_length: int = 2000,  # Increased for longer poems
    num_return_sequences: int = 4,
    temperature: float = 0.7,  # Balanced between creativity and coherence
    top_k: int = 100,  # Increased to allow more diverse vocabulary
    top_p: float = 0.9,  # Higher value for more diverse outputs
    repetition_penalty: float = 1.5,  # Increased to prevent repetition
    early_stopping: bool = False,  # Disabled to allow full length generation
    custom_prompt: Optional[str] = None,
) -> List[str]:
    """
    Generate multiple poetic responses for a given prompt.
    
    Args:
        prompt: Input text to generate from
        max_length: Maximum length of generated text (increased to 2000)
        num_return_sequences: Number of different sequences to generate
        temperature: Controls randomness (0.7 for balanced creativity)
        top_k: Number of highest probability tokens to consider (100 for more diversity)
        top_p: Cumulative probability threshold for token selection (0.9 for more diversity)
        repetition_penalty: Penalty for repeating tokens (1.5 to prevent repetition)
        early_stopping: Whether to stop at the first complete sentence (disabled)
        custom_prompt: Optional custom prompt to guide the generation
    
    Returns:
        List of generated text sequences
    """
    try:
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Process the prompt with custom prompt if provided
        if custom_prompt:
            prompt = f"{custom_prompt}\n\nPrompt: {prompt}"
        
        # Encode input
        inputs = _tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Set up generation parameters
        gen_kwargs = {
            "max_length": max_length,
            "num_return_sequences": num_return_sequences,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": True,
            "pad_token_id": _tokenizer.eos_token_id,
            "no_repeat_ngram_size": 3,  # Prevent repetition of 3-grams
            "length_penalty": 1.2,  # Encourage longer sequences
            "num_beams": 1,  # Use greedy search since we're using temperature sampling
            "min_length": 500,  # Ensure minimum length of 500 tokens
            "eos_token_id": _tokenizer.eos_token_id,
            "pad_token_id": _tokenizer.pad_token_id,
            "bad_words_ids": None,  # No bad words filtering
            "use_cache": True,  # Enable caching for faster generation
            "output_scores": False,  # Don't output scores to save memory
            "return_dict_in_generate": False,  # Return only the sequences
        }

        # Generate sequences
        with torch.no_grad():  # Disable gradient calculation
            output_sequences = _model.generate(inputs, **gen_kwargs)
        
        # Decode and clean up the generated sequences
        generated_sequences = []
        for sequence in output_sequences:
            text = _tokenizer.decode(sequence, skip_special_tokens=True)
            # Remove the input prompt from the output
            text = text[len(_tokenizer.decode(inputs[0], skip_special_tokens=True)):]
            # Clean up the text
            text = text.strip()
            # Ensure the text ends with proper punctuation
            if not text.endswith(('.', '!', '?')):
                text += '.'
            generated_sequences.append(text.strip())
        
        # Clear memory
        del inputs, output_sequences
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return generated_sequences
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        # Clear memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise 