from pathlib import Path
import json
import random

ROOT = Path(__file__).resolve().parents[1]
SFT_DIR = ROOT / "data/processed/sft"
OUT_FILE = SFT_DIR / "dataset_sft_full.jsonl"

FILES = [
    SFT_DIR / "usmle_train_sft.jsonl",
    SFT_DIR / "usmle_dev_sft.jsonl",
    SFT_DIR / "usmle_test_sft.jsonl",
    SFT_DIR / "medmcqa_sft.jsonl",
    SFT_DIR / "us_qbank_sft.jsonl",
]

print("🔄 Cargando datasets...")

records = []

for f in FILES:
    print(f"  ➜ Leyendo {f}")
    with open(f, "r") as fin:
        for line in fin:
            try:
                records.append(json.loads(line))
            except:
                continue

print("📌 Total de ejemplos cargados:", len(records))

print("🔀 Mezclando registros...")
random.shuffle(records)

print(f"💾 Guardando dataset final: {OUT_FILE}")
with open(OUT_FILE, "w") as fout:
    for r in records:
        fout.write(json.dumps(r) + "\n")

print("🎉 Dataset final SFT generado correctamente!")
