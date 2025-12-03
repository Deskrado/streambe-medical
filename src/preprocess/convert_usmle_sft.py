from pathlib import Path
import json

# ROOT ahora apunta a /streambe-medical/src
ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data/raw/data_clean/questions/US"
OUT = ROOT / "data/processed/sft"

print("🔄 Convirtiendo USMLE a SFT...")

OUT.mkdir(parents=True, exist_ok=True)

def convert_file(input_path: Path, output_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"❌ No existe: {input_path}")

    print(f"Convirtiendo: {input_path}")

    # Creamos archivo si no existe
    if not output_path.exists():
        output_path.write_text("")

    with open(input_path, "r") as fin, open(output_path, "a") as fout:
        for line in fin:
            row = json.loads(line)

            question = row.get("question", "").strip()

            options = row.get("options")
            if isinstance(options, list):
                opts_text = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)])
            else:
                opts_text = ""

            context = (
                row.get("long_answer")
                or row.get("context")
                or row.get("extracted_answer")
                or ""
            ).strip()

            answer = (
                row.get("answer")
                or row.get("correct")
                or row.get("label")
                or ""
            ).strip()

            prompt = f"Pregunta: {question}\n\n"
            prompt += f"Contexto: {context}\n\n"
            if opts_text:
                prompt += f"Opciones:\n{opts_text}\n\n"
            prompt += "Respuesta:"

            fout.write(json.dumps({"input": prompt, "output": answer}, ensure_ascii=False) + "\n")

    print(f"✔️ Guardado en {output_path}")


# ---- Archivos USMLE normal ----
convert_file(RAW / "train.jsonl", OUT / "usmle_train_sft.jsonl")
convert_file(RAW / "dev.jsonl", OUT / "usmle_dev_sft.jsonl")
convert_file(RAW / "test.jsonl", OUT / "usmle_test_sft.jsonl")

# ---- Archivos USMLE 4-options ----
FOUR = RAW / "4_options"

convert_file(FOUR / "phrases_no_exclude_train.jsonl", OUT / "usmle_train_sft.jsonl")
convert_file(FOUR / "phrases_no_exclude_dev.jsonl", OUT / "usmle_dev_sft.jsonl")
convert_file(FOUR / "phrases_no_exclude_test.jsonl", OUT / "usmle_test_sft.jsonl")

print("🎉 Conversión completa de USMLE normal + 4-options")
