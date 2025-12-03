from fastapi import FastAPI
from router_inference import router as infer_router
from router_usmle import router as usmle_router
from router_diagnose import router as diagnose_router

app = FastAPI(
    title="Streambe-Medical LLM",
    description="Inference API para modelo médico",
    version="1.0"
)

app.include_router(infer_router)
app.include_router(usmle_router)
app.include_router(diagnose_router)

@app.get("/")
def home():
    return {"status": "ok", "message": "Streambe-Medical LLM API running"}
