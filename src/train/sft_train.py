import os
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import torch


# ==========================================
# CONFIG GENERAL
# ==========================================

cfg = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "tokenized_dataset_path": "data/tokenized/qwen25/sft",
    "output_dir": "checkpoints/sft_qwen25",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "num_train_epochs": 1,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "logging_steps": 10,
    "save_steps": 500,
}


# ==========================================
# FILTRO DE EJEMPLOS VACÍOS (CRÍTICO)
# ==========================================

def has_tokens(example):
    """Filtra registros que no tienen input_ids o están vacíos."""
    return (
        "input_ids" in example
        and example["input_ids"] is not None
        and isinstance(example["input_ids"], list)
        and len(example["input_ids"]) > 0
    )


# ==========================================
# MAIN
# ==========================================

def main():
    print("🔄 Cargando tokenizer y modelo...")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    # ==========================================
    # DATASET
    # ==========================================

    print("📚 Cargando dataset tokenizado...")

    tokenized_path = Path(cfg["tokenized_dataset_path"]).resolve()
    print(f"📁 Path dataset: {tokenized_path}")

    if not tokenized_path.exists():
        raise FileNotFoundError(f"❌ No existe la carpeta: {tokenized_path}")

    dataset = load_from_disk(str(tokenized_path))

    # detecta si es DatasetDict (train + val)
    if hasattr(dataset, "keys"):
        train_data = dataset["train"]
    else:
        train_data = dataset

    print(f"📌 Total muestras cargadas: {len(train_data)}")

    # ==========================================
    # FILTRO DE EXAMPLES VACÍOS
    # ==========================================

    print("🧹 Filtrando ejemplos sin tokens...")
    before = len(train_data)
    train_data = train_data.filter(has_tokens)
    after = len(train_data)
    print(f"✔️ Filtrado vacío: {before} → {after} (eliminados {before - after})")

    if after == 0:
        raise RuntimeError("❌ ERROR: Todos los ejemplos quedaron vacíos. Revisar tokenización.")

    # ==========================================
    # TRAINING CONFIG
    # ==========================================

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_available(),
        optim="adamw_torch",
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=data_collator,
    )

    # ==========================================
    # TRAIN
    # ==========================================

    print("🚀 Iniciando entrenamiento SFT...")
    trainer.train()
    print("🎉 Entrenamiento finalizado correctamente")


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":
    main()
