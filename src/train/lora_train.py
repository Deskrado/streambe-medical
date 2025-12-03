import yaml
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


def load_config(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config("configs/lora_qwen3.yaml")

    print("🔄 Cargando modelo base...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    print("🔧 Aplicando LoRA...")
    lora_cfg = LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        task_type="CAUSAL_LM",
        bias="none",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("📚 Cargando dataset tokenizado...")
    train_data = load_from_disk(cfg["tokenized_dataset_path"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_acc"],
        learning_rate=cfg["learning_rate"],
        logging_steps=20,
        save_steps=500,
        report_to=["wandb"],
        fp16=True,
        gradient_checkpointing=True,
        save_total_limit=2,
        evaluation_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=data_collator,
    )

    print("🚀 Entrenando modelo LoRA...")
    trainer.train()
    print("🎉 Fine-tuning LoRA completo!")


if __name__ == "__main__":
    main()

