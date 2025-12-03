from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset
from pathlib import Path

ROOT = Path("/home/deskrado/streambe-medical")
TOKENIZED_DIR = ROOT / "data/tokenized/qwen25"
SFT_PATH = ROOT / "data/processed/sft/dataset_sft_full.jsonl"
BOOKS_PATH = ROOT / "data/processed/pretrain/textbooks_en.txt"

TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)

print("🔄 Cargando tokenizer Qwen2.5...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# ---------------------------
# CLEAN FUNCTION
# ---------------------------

def clean_text(example):
    txt = example.get("text")

    if txt is None:
        return {"text": ""}

    if not isinstance(txt, str):
        try:
            txt = str(txt)
        except:
            return {"text": ""}

    txt = txt.strip()
    return {"text": txt}

# ---------------------------
# 1. Tokenizar SFT
# ---------------------------

print("📚 Cargando dataset SFT...")

sft_data = load_dataset(
    "json",
    data_files=str(SFT_PATH),
    split="train"
)

print("🧹 Limpiando dataset SFT...")
sft_data = sft_data.map(clean_text)

print("📝 Registros SFT:", len(sft_data))

def tokenize_fn(batch):
    texts = [t if isinstance(t, str) and len(t) > 0 else "" for t in batch["text"]]
    return tok(
        texts,
        truncation=True,
        max_length=2048,
    )

print("🔧 Tokenizando SFT...")
tokenized_sft = sft_data.map(tokenize_fn, batched=True)

print(f"💾 Guardando tokenized SFT en {TOKENIZED_DIR}/sft")
tokenized_sft.save_to_disk(str(TOKENIZED_DIR / "sft"))

# ---------------------------
# 2. Tokenizar libros médicos
# ---------------------------

print("📚 Cargando libros médicos...")

with open(BOOKS_PATH, "r") as f:
    lines = [l.strip() for l in f if len(l.strip()) > 0]

books_dataset = Dataset.from_dict({"text": lines})

print("🔧 Tokenizando pretraining corpus...")
tokenized_books = books_dataset.map(tokenize_fn, batched=True)

print(f"💾 Guardando tokenized pretraining corpus en {TOKENIZED_DIR}/pretrain")
tokenized_books.save_to_disk(str(TOKENIZED_DIR / "pretrain"))

print("🎉 TOKENIZACIÓN COMPLETA — Qwen2.5 LISTO PARA ENTRENAR")
