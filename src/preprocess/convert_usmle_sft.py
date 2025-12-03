import json
from pathlib import Path

# Ruta donde estás parado ahora: ~/streambe-medical/data/raw/medqa_usmle/data_clean/questions/US
BASE = Path(".")

# Ruta de salida absoluta segura
OUT = Path("../../../../../processed/sft")
OUT.mkdir(parents=True, exist_ok=True)

def convert_file(input_path, output_file, four_options=False):
    input_path = Path(input_path)
    output_path = OUT / output_file

    print(f"Convirtiendo: {input_path}")

    with open(input_path, "r") as fin, open(output_path, "a") as fout:
        for line in fin:
            row = json.loads(line)

            if four_options:
                q = row.get("question", "")
                opts = row.get("options", [])
                ans = row.get("answer", "")
            else:
                q = row.get("question", "")
                opts = row.get("options", [])
                ans = row.get("answer", "") or row.get("cop", "")

            prompt = (
                "You are a medical expert.\n"
                f"Question: {q}\n"
                f"Options: {opts}\n"
                "Provide the correct answer.\n"
            )

            output = f"The correct answer is: {ans}"

            fout.write(json.dumps({"input": prompt, "output": output}) + "\n")

    print(f"✔️ Guardado en {output_path}")


# === USMLE NORMAL ===
convert_file("train.jsonl", "usmle_train_sft.jsonl")
convert_file("dev.jsonl",   "usmle_dev_sft.jsonl")
convert_file("test.jsonl",  "usmle_test_sft.jsonl")

# === USMLE 4-OPTIONS CURADO ===
FOUR = Path("4_options")

convert_file(FOUR / "phrases_no_exclude_train.jsonl",
             "usmle_train_sft.jsonl",
             four_options=True)

convert_file(FOUR / "phrases_no_exclude_dev.jsonl",
             "usmle_dev_sft.jsonl",
             four_options=True)

convert_file(FOUR / "phrases_no_exclude_test.jsonl",
             "usmle_test_sft.jsonl",
             four_options=True)

print("🎉 Conversión completa de USMLE normal + 4-options")
