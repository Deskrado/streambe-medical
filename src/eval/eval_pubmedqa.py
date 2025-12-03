import json
from eval_utils import chat, load_model

def evaluate_pubmedqa(model_path, json_path, limit=200):
    tokenizer, model = load_model(model_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    data = data[:limit]

    correct = 0

    for item in data:
        prompt = f"""
Eres un médico investigador. Responde YES, NO o MAYBE.

Contexto:
{item['context']}

Pregunta:
{item['question']}

Respuesta:
"""

        out = chat(model, tokenizer, prompt)
        pred = out.strip().upper()

        if item["answer"] in pred:
            correct += 1

    acc = correct / len(data)
    print(f"📚 PubMedQA Accuracy: {acc:.4f}")
    return acc
