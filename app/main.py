from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router
from app.config import DATASET_ATIVO, MODEL_PATH

# Inicializa a aplicação FastAPI com metadados profissionais para o Swagger
app = FastAPI(
    title="API de Visão Computacional - YOLOv8s",
    description="Microsserviço backend-only de alta performance para detecção e localização de objetos.",
    version="1.0.0"
)

# Configuração de CORS para permitir requisições seguras entre Front-end e Back-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite chamadas de qualquer origem local (como o Streamlit)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos HTTP (POST, GET, etc.)
    allow_headers=["*"],
)

# Inclui o roteador de endpoints sob a raiz da API
app.include_router(api_router)

@app.get("/", summary="Endpoint de verificação de integridade (Healthcheck)")
def health_check():
    """Retorna o status atual do microsserviço e qual inteligência artificial está carregada."""
    return {
        "status": "healthy",
        "dataset_ativo": DATASET_ATIVO,
        "modelo_carregado": MODEL_PATH
    }

if __name__ == "__main__":
    import uvicorn
    # Permite iniciar o servidor executando o arquivo diretamente via Python se necessário
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
