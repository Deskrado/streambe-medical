import json
from eval_utils import load_model, generate_answer

def evaluate_pubmedqa(model_path, dataset):
    model, tokenizer = load_model(model_path)

    correct = 0
    total = 0

    with open(dataset) as f:
        for line in f:
            item = json.loads(line)
            prompt = f"Pregunta: {item['question']}\nContexto: {item['context']}\nRespuesta:"
            pred = generate_answer(model, tokenizer, prompt)

            if item["final_answer"].lower() in pred.lower():
                correct += 1
            total += 1

    acc = correct / total
    print(f"PubMedQA Accuracy: {acc:.4f}")
    return acc
