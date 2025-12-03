import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

# Más adelante podés cambiar a Qwen 3 cuando esté en HF con ese nombre
MODEL_ID = "Qwen/Qwen2-7B"

def tokenize_file(input_path: str, output_path: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open() as f_in, output_path.open("w") as f_out:
        for line in tqdm(f_in, desc=f"Tokenizing {input_path.name}"):
            obj = json.loads(line)
            text = obj.get("text", "")
            enc = tokenizer(
                text,
                truncation=False,
                add_special_tokens=False
            )
            out = {
                "input_ids": enc["input_ids"]
            }
            f_out.write(json.dumps(out) + "\n")

if __name__ == "__main__":
    tokenize_file(
        "data/clean/pubmed_sample.clean.jsonl",
        "data/tokenized/pubmed_sample.tok.jsonl"
    )
