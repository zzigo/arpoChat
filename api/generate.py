import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import List
import logging
from huggingface_hub import HfApi
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise EnvironmentError("HF_TOKEN environment variable is not set")

api = HfApi(token=hf_token)
MODEL_NAME = "animaratio/arpochat"
MODEL_PATH = "model.pt"
CACHE_DIR = Path("models")
CACHE_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
_model = GPT2LMHeadModel.from_pretrained("gpt2")
_model.eval()

def load_model():
    global _model, _tokenizer
    try:
        model_path = CACHE_DIR / MODEL_PATH
        if not model_path.exists():
            logger.info(f"Downloading model from {MODEL_NAME}...")
            api.hf_hub_download(
                repo_id=MODEL_NAME,
                filename=MODEL_PATH,
                local_dir=CACHE_DIR,
                local_dir_use_symlinks=False
            )
        state_dict = torch.load(model_path, map_location=device)
        _model.load_state_dict(state_dict, strict=False)
        _model.to(device)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

load_model()

def is_phonetic(text: str) -> bool:
    """Heuristic to detect phonetic/distorted text."""
    # Look for repeated vowels/consonants or lack of standard words
    return bool(re.search(r'([aeiou]{3,}|\b\w{1,2}\b)', text.lower())) and len(text.split()) < 5

def detect_language_shift(prev_word: str, curr_word: str) -> bool:
    """Simple heuristic for language shift (Spanish vs. other)."""
    spanish_common = {'la', 'el', 'de', 'y', 'en', 'que', 'con', 'del'}
    return (prev_word in spanish_common and curr_word not in spanish_common) or \
           (prev_word not in spanish_common and curr_word in spanish_common)

def process_text(text: str) -> str:
    """Apply punctuation and language rules."""
    words = text.split()
    lines = []
    current_line = []
    prev_word = ""

    for word in words:
        current_line.append(word)
        if prev_word and detect_language_shift(prev_word, word):
            lines.append(" ".join(current_line[:-1]) + ".")
            current_line = [word]
        prev_word = word

    if current_line:
        lines.append(" ".join(current_line))

    # Punctuation enhancement
    processed_lines = []
    for line in lines:
        if is_phonetic(line):
            # Distort punctuation for phonetic text
            line = re.sub(r'[.!?]', lambda m: m.group(0) * 2 if torch.rand(1).item() > 0.5 else "", line)
        else:
            # Add rhythm with punctuation
            line = line.strip()
            if len(line.split()) > 3:
                line = re.sub(r'\b(\w+)\s(\w+)', r'\1, \2', line, count=1)  # Add comma
                if not line.endswith(('.', '!', '?', ':')):
                    line += '.' if torch.rand(1).item() > 0.3 else '!'
            if torch.rand(1).item() > 0.7:  # Randomly add flair
                line = line.replace(" ", " : ", 1) if len(line.split()) > 5 else line + "?"
        processed_lines.append(line)

    return "\n".join(processed_lines)

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
            
            generated_text = raw_text[len(poetic_prompt):].strip() if len(raw_text) > len(poetic_prompt) else raw_text.strip()
            if not generated_text or len(generated_text.split()) < 50:
                logger.warning(f"Short text for temp {temp}: {len(generated_text.split())} words")
                generated_text += "\n" + " ".join([f"{prompt.split()[0]} torcido {i}" for i in range(50)])
            
            # Apply new rules
            generated_text = process_text(generated_text)
            logger.info(f"Processed text for temp {temp}: {generated_text[:100]}...")
            results.append({"text": generated_text, "temperature": temp})
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise

def get_model_info():
    return {
        "name": "ArPoChat",
        "device": str(device),
        "vocab_size": len(_tokenizer),
        "total_params": sum(p.numel() for p in _model.parameters())
    }