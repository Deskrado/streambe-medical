from fastapi import APIRouter
from model_loader import infer

router = APIRouter()

@router.post("/usmle")
def usmle(payload: dict):
    question = payload["question"]
    choices = payload["choices"]

    prompt = f"Pregunta USMLE: {question}\nOpciones: {choices}\nRespuesta:"
    result = infer(prompt)
    
    return {"answer": result}
