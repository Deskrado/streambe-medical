import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer


def generate_answer(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, temperature=0.0
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text[len(prompt):].strip()
