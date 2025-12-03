from pathlib import Path
import json

# Ruta raíz absoluta del proyecto (dinámica, portable)
ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data/raw/medqa_usmle/data_clean/questions/US"
OUT = ROOT / "data/processed/sft"

print("🔄 Convirtiendo USMLE a SFT...")

OUT.mkdir(parents=True, exist_ok=True)

def convert_file(input_path: Path, output_path: Path):
    print(f"Convirtiendo: {input_path}")

    # Escritura en modo "w" para evitar duplicados con los 4-options
    with open(input_path, "r") as fin, open(output_path, "a") as fout:

        for line in fin:
            row = json.loads(line)

            # Campos no uniformes: el dataset cambia entre archivos
            question = row.get("question", "").strip()

            # Opciones si existen (algunos con 4-options, otros no)
            options = row.get("options", None)
            if isinstance(options, list):
                opts_text = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)])
            else:
                opts_text = ""

            # Contexto variable según tipo de archivo
            context = (
                row.get("long_answer")
                or row.get("context")
                or row.get("extracted_answer")
                or ""
            ).strip()

            # Respuestas cambian según origen
            answer = (
                row.get("answer")
                or row.get("correct")
                or row.get("label")
                or ""
            ).strip()

            # Construcción del SFT "input/output"
            full_prompt = (
                f"Pregunta: {question}\n\n"
                f"Contexto: {context}\n\n"
            )

            if opts_text:
                full_prompt += f"Opciones:\n{opts_text}\n\n"

            full_prompt += "Respuesta:"

            fout.write(json.dumps({"input": full_prompt, "output": answer}, ensure_ascii=False) + "\n")

    print(f"✔️ Guardado en {output_path}")


# ---- ARCHIVOS PRINCIPALES ----
convert_file(RAW / "train.jsonl", OUT / "usmle_train_sft.jsonl")
convert_file(RAW / "dev.jsonl", OUT / "usmle_dev_sft.jsonl")
convert_file(RAW / "test.jsonl", OUT / "usmle_test_sft.jsonl")

# ---- ARCHIVOS 4-options ----
FOUR = RAW / "4_options"

convert_file(FOUR / "phrases_no_exclude_train.jsonl", OUT / "usmle_train_sft.jsonl")
convert_file(FOUR / "phrases_no_exclude_dev.jsonl", OUT / "usmle_dev_sft.jsonl")
convert_file(FOUR / "phrases_no_exclude_test.jsonl", OUT / "usmle_test_sft.jsonl")

print("🎉 Conversión completa de USMLE normal + 4-options")
