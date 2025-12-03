import json
from eval_utils import chat, load_model

def load_medqa(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
    return data

def evaluate_medqa(model_path, medqa_path, limit=200):
    tokenizer, model = load_model(model_path)

    data = load_medqa(medqa_path)
    data = data[:limit]

    correct = 0
    total = len(data)

    for i, item in enumerate(data):
        question = item["question"]
        options = item["options"]
        answer = item["answer"]

        prompt = f"""
Eres un médico experto en USMLE. Responde solo con la letra correcta.

Pregunta:
{question}

Opciones:
A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}

Respuesta:
"""

        out = chat(model, tokenizer, prompt)
        pred = out.strip()[-1].upper()

        if pred == answer:
            correct += 1

    acc = correct / total
    print(f"🏥 MedQA Accuracy: {acc:.4f} ({correct}/{total})")
    return acc
