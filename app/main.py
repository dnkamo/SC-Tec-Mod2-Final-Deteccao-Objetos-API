from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router
from app.config import DISPOSITIVO_ATIVO, EXTENSAO_MODELO

app = FastAPI(
    title="API de Visão Computacional Multicenários - YOLOv8s",
    description="Microsserviço dinâmico backend-only com suporte a múltiplos modelos e fallback automático.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/", summary="Endpoint de verificação de integridade (Healthcheck)")
def health_check():
    """Retorna o status atual do microsserviço e especificações do hardware."""
    return {
        "status": "healthy",
        "hardware_detectado": str(DISPOSITIVO_ATIVO),
        "formato_arquivos": EXTENSAO_MODELO,
        "modos_suportados": ["padrao", "placas", "ingredientes"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
