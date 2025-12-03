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

# =======================
# CONFIGURACIÓN GENERAL
# =======================

cfg = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "tokenized_dataset_path": "data/tokenized/qwen25/sft",  # <<<<<< RUTA CORRECTA
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

def main():
    print("🔄 Cargando tokenizer y modelo...")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    # =======================
    # DATASET
    # =======================
    print("📚 Cargando dataset tokenizado...")

    tokenized_path = Path(cfg["tokenized_dataset_path"]).resolve()
    print(f"📁 Path dataset: {tokenized_path}")

    if not tokenized_path.exists():
        raise FileNotFoundError(f"❌ No existe la carpeta: {tokenized_path}")

    dataset = load_from_disk(str(tokenized_path))

    # Si es DatasetDict (train/val)
    if isinstance(dataset, dict) or hasattr(dataset, "keys"):
        train_data = dataset["train"]
    else:
        train_data = dataset

    print(f"📌 Total muestras: {len(train_data)}")

    # =======================
    # TRAINING CONFIG
    # =======================
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
        bf16=torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
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

    # =======================
    # TRAIN
    # =======================
    print("🚀 Iniciando entrenamiento SFT...")
    trainer.train()
    print("🎉 Entrenamiento finalizado")

if __name__ == "__main__":
    main()
