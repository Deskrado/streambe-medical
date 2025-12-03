from datasets import Dataset
from transformers import AutoTokenizer
from pathlib import Path
import json
import random

PROJECT = Path.home() / "streambe-medical"

SFT_DATA = PROJECT / "data/processed/sft/dataset_sft_full.jsonl"
PRETRAIN_DATA = PROJECT / "data/processed/pretrain/textbooks_en.txt"
OUT_DIR = PROJECT / "data/tokenized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-7B"

print("🔄 Cargando tokenizer Qwen2.5...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# ---------------------------
# SFT DATASET
# ---------------------------

print("📚 Cargando dataset SFT...")
records = []
with open(SFT_DATA, "r") as f:
    for line in f:
        obj = json.loads(line)
        text = f"<input>{obj['input']}\n<output>{obj['output']}"
        records.append({"text": text})

random.shuffle(records)
split = int(len(records) * 0.98)
train_ds = Dataset.from_list(records[:split])
val_ds = Dataset.from_list(records[split:])

print(f"📝 Train SFT: {len(train_ds)}")
print(f"📝 Val SFT: {len(val_ds)}")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=2048,
        padding="max_length"
    )

print("🔧 Tokenizando SFT train...")
train_t = train_ds.map(tokenize, batched=True, remove_columns=["text"])
train_t.save_to_disk(str(OUT_DIR / "sft_train"))

print("🔧 Tokenizando SFT val...")
val_t = val_ds.map(tokenize, batched=True, remove_columns=["text"])
val_t.save_to_disk(str(OUT_DIR / "sft_val"))

# ---------------------------
# PRETRAIN TEXTBOOKS
# ---------------------------

print("📚 Cargando libros médicos...")
with open(PRETRAIN_DATA, "r", errors="ignore") as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]

pre_ds = Dataset.from_list([{"text": l} for l in lines])

def tokenize_pretrain(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=2048,
        padding="max_length"
    )

print("🔧 Tokenizando pretraining corpus...")
pre_t = pre_ds.map(tokenize_pretrain, batched=True, remove_columns=["text"])
pre_t.save_to_disk(str(OUT_DIR / "pretrain"))

print("🎉 TOKENIZACIÓN COMPLETA — Qwen2.5 LISTO PARA ENTRENAR")
