import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk
import os


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config("configs/pretrain_qwen3.yaml")

    print("🔄 Cargando tokenizer y modelo...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    print("📚 Cargando dataset tokenizado...")
    train_data = load_from_disk(cfg["tokenized_dataset_path"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_ratio=cfg["warmup_ratio"],
        learning_rate=cfg["learning_rate"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=2,
        fp16=cfg["fp16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        report_to=cfg["report_to"],
        evaluation_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=data_collator,
    )

    print("🚀 Iniciando PRETRAIN...")
    trainer.train()
    print("🎉 Pretrain finalizado")


if __name__ == "__main__":
    main()

