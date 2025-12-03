from fastapi import APIRouter
from model_loader import infer

router = APIRouter()

@router.post("/predict")
def predict(payload: dict):
    prompt = payload.get("prompt", "")
    result = infer(prompt)
    return {"response": result}
