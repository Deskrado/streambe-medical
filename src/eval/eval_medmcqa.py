import pandas as pd
from eval_utils import chat, load_model

def evaluate_medmcqa(model_path, dataset_path, limit=200):
    tokenizer, model = load_model(model_path)

    df = pd.read_csv(dataset_path).head(limit)

    correct = 0

    for _, row in df.iterrows():
        prompt = f"""
Responde la opción correcta (A/B/C/D).

Pregunta:
{row['question']}

Opciones:
A) {row['opa']}
B) {row['opb']}
C) {row['opc']}
D) {row['opd']}

Respuesta:
        """

        out = chat(model, tokenizer, prompt)
        pred = out.strip()[-1].upper()

        if pred == row["cop"].upper():
            correct += 1

    acc = correct / len(df)
    print(f"🧪 MedMCQA Accuracy: {acc:.4f} ({correct}/{len(df)})")
    return acc
