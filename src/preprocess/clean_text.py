import re
import json
from pathlib import Path
from tqdm import tqdm

def clean_medical_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\r", " ").replace("\n", " ")
    return text.strip()

def process_file(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open() as f_in, output_path.open("w") as f_out:
        for line in tqdm(f_in, desc=f"Cleaning {input_path.name}"):
            obj = json.loads(line)
            text = obj.get("text", "")
            obj["text"] = clean_medical_text(text)
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # Ejemplo: limpiar un archivo raw -> clean
    process_file("data/raw/pubmed_sample.jsonl",
                 "data/clean/pubmed_sample.clean.jsonl")
