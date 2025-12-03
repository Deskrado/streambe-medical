from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEXTBOOKS_DIR = PROJECT_ROOT / "data/raw/data_clean/textbooks/en"
OUT = PROJECT_ROOT / "data/processed/pretrain/textbooks_en.txt"

OUT.parent.mkdir(parents=True, exist_ok=True)

print("📚 Leyendo libros médicos...")

with open(OUT, "w") as fout:
    for txt_file in sorted(TEXTBOOKS_DIR.glob("*.txt")):
        print("  ➜", txt_file.name)

        with open(txt_file, "r", errors="ignore") as fin:
            content = fin.read()

            # Limpieza básica
            content = content.replace("\r", "")
            content = "\n".join([line.strip() for line in content.split("\n") if line.strip()])

            fout.write(content + "\n\n")

print("🎉 Archivo final creado:", OUT)
