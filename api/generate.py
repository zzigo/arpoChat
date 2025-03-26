import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import List
import logging
from pathlib import Path
import psutil  # For memory usage

# Logging to file as specified
logging.basicConfig(level=logging.INFO, filename='/var/log/arpochat/arpochat.log')
logger = logging.getLogger(__name__)

MODEL_PATH = "/opt/arpoChat/models/model.pt"  # Local path to your model
device = torch.device("cpu")
logger.info(f"Using device: {device}")

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # MB
    return mem

# Preload model at startup
logger.info(f"Memory before loading: {get_memory_usage():.2f} MB")
logger.info(f"Loading model from {MODEL_PATH}")
_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
_model = GPT2LMHeadModel.from_pretrained("gpt2")
logger.info(f"Memory after base model: {get_memory_usage():.2f} MB")

model_path = Path(MODEL_PATH)
if not model_path.exists():
    logger.error(f"Model file not found at {MODEL_PATH}")
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

logger.info(f"Memory before state dict: {get_memory_usage():.2f} MB")
state_dict = torch.load(model_path, map_location=device, weights_only=True)
_model.load_state_dict(state_dict)
logger.info(f"Memory after state dict: {get_memory_usage():.2f} MB")

_model.to(device)
if _tokenizer.pad_token is None:
    _tokenizer.pad_token = _tokenizer.eos_token
logger.info("Model loaded successfully")
logger.info(f"Final memory usage: {get_memory_usage():.2f} MB")

def generate_text(prompt: str, temperatures: List[float] = [0.1, 0.5, 0.9], max_length: int = 1024) -> List[dict]:
    try:
        poetic_prompt = f"""Desde "{prompt}", despliega un poema dadaísta argentino: repite estructuras como ecos rotos, 
        teje conexiones semánticas absurdas pero brillantes, y cierra con giros lógicos que desafíen la razón. 
        Invoca imágenes surrealistas—tangos deshechos, pampas torcidas, vanguardias de Buenos Aires—y 
        estira el texto hasta al menos 300 palabras."""
        
        inputs = _tokenizer(poetic_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        logger.info(f"Tokenized input: {inputs['input_ids'].shape}")
        results = []
        for temp in temperatures:
            with torch.no_grad():
                outputs = _model.generate(
                    **inputs,
                    max_new_tokens=600,
                    min_length=inputs['input_ids'].shape[1] + 300,
                    temperature=temp,
                    do_sample=True,
                    top_k=40,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=_tokenizer.pad_token_id,
                    repetition_penalty=1.1
                )
            raw_text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Raw output for temp {temp}: {raw_text[:100]}...")
            generated_text = raw_text[len(poetic_prompt):].strip() or raw_text.strip()
            if len(generated_text.split()) < 50:
                generated_text += "\n" + " ".join([f"{prompt.split()[0]} torcido {i}" for i in range(50)])
            generated_text = "\n".join([line.strip() for line in generated_text.split(".") if line.strip()])
            results.append({"text": generated_text, "temperature": temp})
        return results
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise

def get_model_info():
    try:
        return {
            "name": "ArPoChat",
            "device": str(device),
            "vocab_size": len(_tokenizer),
            "total_params": sum(p.numel() for p in _model.parameters())
        }
    except Exception as e:
        logger.error(f"Error in get_model_info: {str(e)}")
        raise