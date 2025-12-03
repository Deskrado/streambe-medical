import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from rlhf_utils import RewardDataset


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config("configs/reward_model.yaml")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    print("📚 Cargando dataset RM...")
    ds = load_dataset("json", data_files=cfg["reward_dataset"])

    train_set = RewardDataset(ds["train"], tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=1,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_acc"],
        num_train_epochs=cfg["epochs"],
        logging_steps=20,
        save_steps=500,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
    )

    print("🚀 Entrenando Reward Model...")
    trainer.train()
    print("🎉 RM entrenado correctamente!")


if __name__ == "__main__":
    main()

