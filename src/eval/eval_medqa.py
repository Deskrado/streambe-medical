import json
from eval_utils import load_model, generate_answer

def evaluate_medqa(model_path, dataset_path):
    model, tokenizer = load_model(model_path)

    correct = 0
    total = 0

    with open(dataset_path) as f:
        for line in f:
            item = json.loads(line)
            prompt = f"Pregunta: {item['question']}\nOpciones: {item['options']}\nRespuesta:"
            pred = generate_answer(model, tokenizer, prompt)

            if pred.strip().upper().startswith(item["answer"].upper()):
                correct += 1
            total += 1

    acc = correct / total
    print(f"MedQA Accuracy: {acc:.4f}")
    return acc
