from eval_medqa import evaluate_medqa
from eval_medmcqa import evaluate_medmcqa
from eval_pubmedqa import evaluate_pubmedqa
from eval_mimic import evaluate_mimic

MODEL = "checkpoints/qwen25_lora_cpu"

def main():
    print("\n===== 🏥 Evaluación Médica Completa =====\n")

    evaluate_medqa(MODEL, "data/raw/medqa_usmle/data_clean/questions/US/test.jsonl", limit=200)
    evaluate_medmcqa(MODEL, "data/raw/medmcqa/test.csv", limit=200)
    evaluate_pubmedqa(MODEL, "data/raw/pubmedqa/test.json", limit=200)
    evaluate_mimic(MODEL, "data/raw/mimic/test.json", limit=10)

if __name__ == "__main__":
    main()
