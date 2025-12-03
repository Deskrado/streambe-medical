import json
import argparse
from eval_medqa import evaluate_medqa
from eval_medmcqa import evaluate_medmcqa
from eval_pubmedqa import evaluate_pubmedqa
from eval_mimic import evaluate_mimic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Ruta del modelo entrenado")
    parser.add_argument("--out", default="eval_report.json", help="Archivo de salida JSON")
    parser.add_argument("--medqa", default="data/benchmarks/medqa.jsonl")
    parser.add_argument("--medmcqa", default="data/benchmarks/medmcqa.jsonl")
    parser.add_argument("--pubmedqa", default="data/benchmarks/pubmedqa.jsonl")
    parser.add_argument("--mimic", default="data/benchmarks/mimic_qa.json")
    args = parser.parse_args()

    print("=======================================")
    print("  🚀 Streambe-Medical — Benchmark Suite")
    print("=======================================")

    results = {}

    print("\n🩺 Evaluando MedQA (USMLE)...")
    results["MedQA"] = evaluate_medqa(args.model, args.medqa)

    print("\n🧠 Evaluando MedMCQA...")
    results["MedMCQA"] = evaluate_medmcqa(args.model, args.medmcqa)

    print("\n📚 Evaluando PubMedQA...")
    results["PubMedQA"] = evaluate_pubmedqa(args.model, args.pubmedqa)

    print("\n🏥 Evaluando MIMIC-III Clinical QA...")
    results["MIMIC-III"] = evaluate_mimic(args.model, args.mimic)

    print("\n=======================================")
    print("🧾 RESULTADOS")
    print("=======================================")

    for k, v in results.items():
        print(f"{k:12s} → {v:.4f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Reporte guardado en: {args.out}")
    print("=======================================\n")


if __name__ == "__main__":
    main()
