from pathlib import Path
import json

# Rutas absolutas dentro del repo
ROOT = Path("/home/deskrado/streambe-medical")
RAW = ROOT / "data/raw/medqa_usmle/data_clean/questions/US"
OUT = ROOT / "data/processed/sft"

print("🔄 Convirtiendo USMLE a SFT...")

OUT.mkdir(parents=True, exist_ok=True)

def convert_file(input_path: Path, output_file: Path):
    print(f"Convirtiendo: {input_path.name}")
    with open(input_path, "r") as fin, open(output_file, "a") as fout:
        for line in fin:
            row = json.loads(line)

            question = row.get("question", "")
            context = row.get("long_answer", "") or row.get("context", "")
            answer = row.get("answer", "")

            text = (
                f"Pregunta: {question}\n"
                f"Contexto: {context}\n"
                f"Respuesta: {answer}"
            )
            fout.write(json.dumps({"text": text}) + "\n")

            # Clean duplicates on append
    print(f"✔️ Guardado en {output_file}")

# Archivos principales
convert_file(RAW / "train.jsonl", OUT / "usmle_train_sft.jsonl")
convert_file(RAW / "dev.jsonl", OUT / "usmle_dev_sft.jsonl")
convert_file(RAW / "test.jsonl", OUT / "usmle_test_sft.jsonl")

# Archivos 4-options
FOUR = RAW / "4_options"

convert_file(FOUR / "phrases_no_exclude_train.jsonl", OUT / "usmle_train_sft.jsonl")
convert_file(FOUR / "phrases_no_exclude_dev.jsonl", OUT / "usmle_dev_sft.jsonl")
convert_file(FOUR / "phrases_no_exclude_test.jsonl", OUT / "usmle_test_sft.jsonl")

print("🎉 Conversión completa de USMLE normal + 4-options")
