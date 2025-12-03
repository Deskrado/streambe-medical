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
    cfg = load_config("configs/sft_qwen3.yaml")

    print("🔄 Cargando tokenizer y modelo...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print("📚 Cargando dataset tokenizado...")
    train_data = load_from_disk(cfg["tokenized_dataset_path"])
    val_data = load_from_disk(cfg["val_dataset_path"])

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_ratio=cfg["warmup_ratio"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        logging_steps=cfg["logging_steps"],
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
        fp16=cfg["fp16"],
        bf16=cfg["bf16"],
        report_to=cfg["report_to"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        evaluation_strategy="steps",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
    )

    print("🚀 INICIANDO ENTRENAMIENTO SFT...")
    trainer.train()
    print("🎉 Entrenamiento SFT finalizado")


if __name__ == "__main__":
    main()
