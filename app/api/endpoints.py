from typing import Literal
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from app.services.yolo_service import YoloService

router = APIRouter()
yolo_service = YoloService()

@router.post("/detect", summary="Realiza a detecção de objetos mapeando o cenário selecionado")
async def detect_objects(
    file: UploadFile = File(...),
    cenario: Literal["placas", "ingredientes", "padrao"] = Query(
        "padrao", 
        description="Escolha qual inteligência artificial deve processar e analisar esta imagem."
    )
):
    """
    Endpoint HTTP POST que aceita o upload de uma imagem e o cenário de inteligência artificial desejado.
    Se o cenário customizado não for encontrado no servidor, o sistema aplica o Fallback automático para o modelo padrão.
    """
    extensoes_suportadas = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if not file.filename.lower().endswith(extensoes_suportadas):
        raise HTTPException(
            status_code=400, 
            detail=f"Formato de arquivo inválido. Extensões suportadas: {', '.join(extensoes_suportadas)}"
        )
    
    try:
        imagem_bytes = await file.read()
        
        # Dispara o pipeline passando o cenário dinâmico escolhido pelo usuário
        resultado = yolo_service.detectar_objetos(imagem_bytes, cenario=cenario)
        
        if resultado.get("status") == "error":
            raise HTTPException(status_code=422, detail=resultado.get("message"))
            
        return resultado
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erro interno ao processar a inferência no microsserviço: {str(e)}"
        )
