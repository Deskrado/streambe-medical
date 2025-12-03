# Streambe-Medical — Open Medical LLM (Research Project)

Streambe-Medical es un proyecto de investigación que busca crear un modelo
de lenguaje especializado en medicina, basado en **Qwen 3 (7B)** y entrenado
con técnicas modernas: SFT/LoRA, continual-pretraining y RLHF/DPO.

Este repositorio permite:
- Preprocesar datasets médicos open-source
- Entrenar LoRA clínico localmente
- Ejecutar continual pretraining en RunPod/Paperspace
- Aplicar RLHF para mejorar seguridad y razonamiento médico
- Evaluar desempeño en MedQA, PubMedQA y otros benchmarks

⚠️ **Advertencia:** Este proyecto es únicamente con fines de investigación.
No está aprobado para uso clínico real.

---

## 🚀 Objetivos

1. Crear un modelo especializado en razonamiento médico.
2. Reducir alucinaciones mediante RLHF.
3. Proveer un pipeline reproducible, escalable y simple.
4. Ofrecer una base open-source para futuros modelos médicos.

---

## 🏗️ Arquitectura del Proyecto

- `/data` → datasets raw, procesados, tokenizados, splits, RLHF.
- `/src/preprocess` → limpieza, tokenización, splits.
- `/src/training` → LoRA, continual pretraining, RLHF.
- `/src/evaluation` → métricas y tests.
- `/src/infer` → servidor de inferencia.
- `/configs` → JSON config para cada entrenamiento.
- `/models` → checkpoints organizados.
- `/notebooks` → exploración y generación RLHF.

---

## 🧪 Entrenamientos disponibles

### 1. LoRA (local, GPU doméstica)

bash scripts/run_lora.sh

### 2. Continual Pretraining (RunPod/Paperspace)

bash scripts/run_continual.sh

### 3. RLHF (DPO) seguro médico

bash scripts/run_rlhf.sh

### 4. Evaluación

bash scripts/run_eval.sh

---

## 📦 Inferencia
Servidor FastAPI/vLLM:

python src/infer/inference_server.py

---

## 📜 Licencia
MIT — con restricción ética: no para uso clínico sin aprobación regulatoria.
