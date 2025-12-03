import yaml
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config("configs/lora_cpu_qwen3.yaml")

    print("🔄 Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)

    print("🧠 Cargando modelo base en CPU (sin bitsandbytes)...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        device_map={"": "cpu"},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print("✨ Aplicando LoRA (CPU-friendly)...")
    lora_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        task_type="CAUSAL_LM",
        bias="none",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("📚 Cargando dataset tokenizado...")
    train_data = load_from_disk(cfg["tokenized_dataset_path"])
    val_data = load_from_disk(cfg["val_dataset_path"])

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_acc"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=2,
        evaluation_strategy="steps",
        fp16=False,  # CPU only
        bf16=False,
        gradient_checkpointing=True,
        report_to=cfg["report_to"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
    )

    print("🚀 Entrenando modelo (LoRA CPU)...")
    trainer.train()
    print("🎉 Entrenamiento terminado!")


if __name__ == "__main__":
    main()

