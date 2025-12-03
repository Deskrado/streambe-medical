import json
from pathlib import Path

PROJECT_ROOT = Path.home() / "streambe-medical"

RAW = PROJECT_ROOT / "data/raw/medqa_usmle/data_clean/questions/US/US_qbank.jsonl"
OUT = PROJECT_ROOT / "data/processed/sft/us_qbank_sft.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

print("🔄 Convirtiendo US_QBANK a SFT...")

with open(RAW, "r") as fin, open(OUT, "w") as fout:
    for line in fin:
        row = json.loads(line)

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

print("🎉 US_QBANK convertido correctamente en:", OUT)
