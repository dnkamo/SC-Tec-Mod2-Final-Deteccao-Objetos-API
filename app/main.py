from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router
from app.config import DISPOSITIVO_ATIVO, EXTENSAO_MODELO

# Inicializa a aplicação FastAPI com metadados profissionais para o Swagger
app = FastAPI(
    title="API de Visão Computacional Multicenários - YOLOv8s",
    description="Microsserviço dinâmico backend-only com suporte a múltiplos modelos e fallback automático.",
    version="1.1.0"
)

# Origens permitidas mapeadas cirurgicamente (Bypass definitivo de barreira local)
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:8501",    # Porta padrão do seu Streamlit
    "http://127.0.0.1:8501",
]

# Configuração de CORS para permitir requisições seguras entre Front-end e Back-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Injeta as origens locais explícitas
    allow_credentials=True,
    allow_methods=["*"],         # Permite todos os métodos HTTP (POST, GET, etc.)
    allow_headers=["*"],
)

# Inclui o roteador de endpoints sob a raiz da API
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
    # Correção do escopo de carregamento de string de módulo
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
