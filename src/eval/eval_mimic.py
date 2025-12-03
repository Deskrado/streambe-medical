import json
from eval_utils import chat, load_model

def evaluate_mimic(model_path, mimic_json_path, limit=20):
    tokenizer, model = load_model(model_path)

    with open(mimic_json_path, "r") as f:
        data = json.load(f)

    data = data[:limit]

    for i, note in enumerate(data):
        prompt = f"""
Eres un clínico. Resume la nota y da un diagnóstico probable.

Nota clínica:
{note['text']}

Responde en formato:
- Resumen:
- Diagnóstico probable:
- Plan sugerido:
"""

        out = chat(model, tokenizer, prompt)
        print(f"\n======== Caso {i+1} ========\n")
        print(out)
