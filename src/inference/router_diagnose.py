from fastapi import APIRouter
from model_loader import infer

router = APIRouter()

@router.post("/diagnose")
def diagnose(payload: dict):
    symptoms = payload["symptoms"]
    context = payload.get("context", "")

    prompt = (
        "Eres un asistente médico experto. "
        f"Paciente presenta: {symptoms}. "
    )
    if context:
        prompt += f"Antecedentes: {context}. "

    prompt += "Diagnóstico probable y conducta clínica:"
    result = infer(prompt, max_new_tokens=300)

    return {"diagnosis": result}
