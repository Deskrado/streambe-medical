import torch
from torch.utils.data import Dataset


class RewardDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        chosen = self.tokenizer(
            item["prompt"] + item["chosen"],
            truncation=True, max_length=self.max, return_tensors="pt"
        )

        rejected = self.tokenizer(
            item["prompt"] + item["rejected"],
            truncation=True, max_length=self.max, return_tensors="pt"
        )

        return {
            "chosen_input_ids": chosen["input_ids"][0],
            "rejected_input_ids": rejected["input_ids"][0],
        }


def build_ppo_dataset(dataset, tokenizer, max_length=512):
    prompts = []
    for item in dataset:
        enc = tokenizer(
            item["prompt"],
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        prompts.append({"input_ids": enc["input_ids"][0]})
    return prompts
