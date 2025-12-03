import yaml
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
from rlhf_utils import build_ppo_dataset


def load_config(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config("configs/ppo.yaml")

    print("🔄 Cargando PPO Model...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg["model_name"],
        device_map="auto",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    print("📚 Cargando dataset PPO...")
    ds = load_dataset("json", data_files=cfg["ppo_dataset"])
    ppo_data = build_ppo_dataset(ds["train"], tokenizer)

    ppo_config = PPOConfig(
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        mini_batch_size=cfg["mini_batch"],
        gradient_accumulation_steps=cfg["grad_acc"],
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=ppo_data,
    )

    print("🚀 Entrenando con PPO...")
    trainer.train()
    print("🎉 PPO completado!")


if __name__ == "__main__":
    main()
