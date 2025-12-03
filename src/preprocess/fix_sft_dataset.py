import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src/data/processed/sft/dataset_sft_full.jsonl"
DST = ROOT / "src/data/processed/sft/dataset_sft_full_clean.jsonl"

print("🔎 Reparando dataset SFT...")

ok = 0
skipped = 0

with open(SRC, "r") as fin, open(DST, "w") as fout:
    for i, line in enumerate(fin, 1):
        line = line.strip()

        if not line:
            skipped += 1
            continue

        try:
            obj = json.loads(line)
        except Exception:
            print(f"❌ Línea corrupta en {i}, saltando...")
            skipped += 1
            continue

        # Forzar que output sea SIEMPRE string
        output = obj.get("output", "")
        if not isinstance(output, str):
            output = str(output)

        obj["output"] = output

        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        ok += 1

print("🎉 Dataset reparado")
print(f"✔️ Líneas correctas: {ok}")
print(f"⚠️ Líneas eliminadas: {skipped}")
print(f"📁 Guardado en: {DST}")
