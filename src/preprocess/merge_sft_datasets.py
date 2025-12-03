import json
import random
from pathlib import Path

PROJECT_ROOT = Path.home() / "streambe-medical"
SFT_DIR = PROJECT_ROOT / "data/processed/sft"
OUT = PROJECT_ROOT / "data/processed/sft/dataset_sft_full.jsonl"

FILES = [
    "usmle_train_sft.jsonl",
    "usmle_dev_sft.jsonl",
    "usmle_test_sft.jsonl",
    "medmcqa_sft.jsonl",
    "us_qbank_sft.jsonl",
]

print("🔄 Cargando datasets...")

records = []

for fname in FILES:
    path = SFT_DIR / fname
    print(f"  ➜ Leyendo {path}")

    with open(path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "input" in obj and "output" in obj:
                    records.append(obj)
            except:
                continue

print(f"📌 Total de ejemplos cargados: {len(records)}")

print("🔀 Mezclando registros...")
random.shuffle(records)

print("💾 Guardando dataset final:", OUT)
with open(OUT, "w") as fout:
    for r in records:
        fout.write(json.dumps(r) + "\n")

print("🎉 Dataset final SFT generado correctamente!")
