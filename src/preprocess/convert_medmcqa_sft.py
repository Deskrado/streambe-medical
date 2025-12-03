import pandas as pd
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]       # /streambe-medical/
RAW = BASE / "data/raw/medmcqa"
OUT = BASE / "data/processed/sft"

print("🔄 Convirtiendo MedMCQA a SFT...")

OUT.mkdir(parents=True, exist_ok=True)

train_path = RAW / "train.csv"
valid_path = RAW / "valid.csv"
test_path = RAW / "test.csv"

if not train_path.exists():
    raise FileNotFoundError(f"❌ No se encontró {train_path}")

df_train = pd.read_csv(train_path)
df_valid = pd.read_csv(valid_path)
df_test = pd.read_csv(test_path)

def convert(df):
    records = []
    for _, row in df.iterrows():
        question = row["question"]
        choices = "\n".join([
            f"A) {row['opa']}",
            f"B) {row['opb']}",
            f"C) {row['opc']}",
            f"D) {row['opd']}",
        ])
        input_text = f"{question}\n{choices}"
        output = row["cop"]

        records.append({
            "input": input_text,
            "output": output
        })
    return records

all_data = (
    convert(df_train) +
    convert(df_valid) +
    convert(df_test)
)

output_file = OUT / "medmcqa_sft.jsonl"
with open(output_file, "w") as f:
    for item in all_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"🎉 MedMCQA convertido correctamente en: {output_file}")
