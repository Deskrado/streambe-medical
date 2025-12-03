import json
import pandas as pd
from pathlib import Path

# Ruta raíz del proyecto
PROJECT_ROOT = Path.home() / "streambe-medical"

# Ruta del dataset MedMCQA
RAW = PROJECT_ROOT / "data/raw/medmcqa"

# Ruta de salida SFT
OUT = PROJECT_ROOT / "data/processed/sft/medmcqa_sft.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

print("🔄 Convirtiendo MedMCQA a SFT...")

df = pd.read_csv(RAW / "train.csv")

with open(OUT, "w") as fout:
    for _, row in df.iterrows():
        q = row["question"]
        opts = [row["opa"], row["opb"], row["opc"], row["opd"]]
        ans = row["cop"]

        prompt = (
            "You are a medical expert.\n"
            f"Question: {q}\n"
            f"Options: {opts}\n"
            "Provide the correct answer.\n"
        )

        output = f"The correct answer is: {ans}"

        fout.write(json.dumps({"input": prompt, "output": output}) + "\n")

print("🎉 MedMCQA convertido correctamente en:", OUT)
