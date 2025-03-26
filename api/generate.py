import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import List
import logging
from huggingface_hub import HfApi
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise EnvironmentError("HF_TOKEN environment variable is not set")

api = HfApi(token=hf_token)
MODEL_NAME = "animaratio/arpochat"  # Replace with your new <500MB model after training
MODEL_PATH = "model.pt"
CACHE_DIR = Path("models")
CACHE_DIR.mkdir(exist_ok=True)

device = torch.device("cpu")  # Render free tier has no GPU
logger.info(f"Using device: {device}")

# Lazy loading
_tokenizer = None
_model = None

def load_model():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        logger.info(f"Loading model from {MODEL_NAME}")
        _tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        _model = GPT2LMHeadModel.from_pretrained("gpt2", low_cpu_mem_usage=True)
        model_path = CACHE_DIR / MODEL_PATH
        if not model_path.exists():
            api.hf_hub_download(repo_id=MODEL_NAME, filename=MODEL_PATH, local_dir=CACHE_DIR)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        _model.load_state_dict(state_dict)
        _model.to(device)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        torch.cuda.empty_cache()  # Clear any residual GPU memory (though CPU-only here)
        logger.info("Model loaded successfully")

def generate_text(prompt: str, temperatures: List[float] = [0.1, 0.5, 0.9], max_length: int = 1024) -> List[dict]:
    try:
        load_model()  # Load only when needed
        poetic_prompt = f"""Desde "{prompt}", despliega un poema dadaísta argentino: repite estructuras como ecos rotos, 
        teje conexiones semánticas absurdas pero brillantes, y cierra con giros lógicos que desafíen la razón. 
        Invoca imágenes surrealistas—tangos deshechos, pampas torcidas, vanguardias de Buenos Aires—y 
        estira el texto hasta al menos 300 palabras."""
        
        inputs = _tokenizer(poetic_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
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
    load_model()
    return {
        "name": "ArPoChat",
        "device": str(device),
        "vocab_size": len(_tokenizer),
        "total_params": sum(p.numel() for p in _model.parameters())
    }