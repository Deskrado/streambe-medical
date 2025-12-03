from eval_utils import load_model, generate_answer
import random
import json

def evaluate_mimic(model_path, mimic_dataset):
    model, tokenizer = load_model(model_path)

    with open(mimic_dataset) as f:
        data = json.load(f)

    sample = random.sample(data, 200)

    score = 0
    for item in sample:
        context = item["note"]
        question = item["question"]
        true = item["answer"]

        prompt = f"Historia clínica: {context}\nPregunta: {question}\nRespuesta:"
        pred = generate_answer(model, tokenizer, prompt)

        if true.lower() in pred.lower():
            score += 1

    acc = score / len(sample)
    print(f"MIMIC-III QA Accuracy: {acc:.4f}")
    return acc
