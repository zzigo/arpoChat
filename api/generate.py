import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import List, Dict, Any, Optional
import logging
from huggingface_hub import HfApi
import gc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get HF token from environment
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise EnvironmentError("HF_TOKEN environment variable is not set")

# Initialize Hugging Face API
api = HfApi(token=hf_token)

# Initialize tokenizer and model with token
try:
    logger.info("Initializing tokenizer and model...")
    # First verify token is valid
    api.whoami()
    
    # Initialize with authentication
    _tokenizer = GPT2Tokenizer.from_pretrained(
        'gpt2',
        use_auth_token=hf_token,
        local_files_only=False  # Force download from HF
    )
    
    # Load model with memory optimizations
    _model = GPT2LMHeadModel.from_pretrained(
        'gpt2',
        use_auth_token=hf_token,
        local_files_only=False,  # Force download from HF
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32  # Use float32 instead of float16 for better compatibility
    )
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(device)
    
    # Enable evaluation mode
    _model.eval()
    
    logger.info("Model and tokenizer initialized successfully")
except Exception as e:
    logger.error(f"Error initializing model or tokenizer: {str(e)}")
    raise

def get_model():
    """Return the loaded model."""
    return _model

def generate_text(
    prompt: str,
    max_length: int = 100,
    num_return_sequences: int = 3,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    early_stopping: bool = True,
) -> List[str]:
    """
    Generate multiple poetic responses for a given prompt.
    
    Args:
        prompt: Input text to generate from
        max_length: Maximum length of generated text
        num_return_sequences: Number of different sequences to generate
        temperature: Higher values = more creative, lower = more focused
        top_k: Number of highest probability tokens to consider
        top_p: Cumulative probability threshold for token selection
        repetition_penalty: Penalty for repeating tokens
        early_stopping: Whether to stop at the first complete sentence
    
    Returns:
        List of generated text sequences
    """
    try:
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
            "early_stopping": early_stopping,
            "pad_token_id": _tokenizer.eos_token_id,
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