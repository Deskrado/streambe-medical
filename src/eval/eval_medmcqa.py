import json
from eval_utils import load_model, generate_answer

def evaluate_medmcqa(model_path, dataset_path):
    model, tokenizer = load_model(model_path)

    correct = 0
    total = 0

    with open(dataset_path) as f:
        for line in f:
            row = json.loads(line)
            q = row["question"]
            choices = row["options"]
            ans = row["cop"]

            prompt = f"Pregunta: {q}\nOpciones: {choices}\nRespuesta:"
            pred = generate_answer(model, tokenizer, prompt)

            if ans.lower() in pred.lower():
                correct += 1
            total += 1

    acc = correct / total
    print(f"MedMCQA Accuracy: {acc:.4f}")
    return acc
