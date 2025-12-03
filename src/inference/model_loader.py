from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

_model = None
_tokenizer = None

def load_model(model_path):
    global _model, _tokenizer

    if _model is None:
        print("🔄 Cargando modelo Streambe-Medical...")
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ Modelo cargado y listo")
    return _model, _tokenizer


def infer(prompt, max_new_tokens=256):
    model, tokenizer = load_model("output/qlora_qwen3")  # default path

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result[len(prompt):].strip()
