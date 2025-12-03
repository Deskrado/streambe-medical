import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path, dtype=torch.float32):
    print(f"🔄 Cargando modelo: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )
    return tokenizer, model

def chat(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
