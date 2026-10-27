from fastapi import APIRouter, File, UploadFile, HTTPException
from app.services.yolo_service import YoloService

# Inicializa o roteador do FastAPI para agrupar as rotas da API
router = APIRouter()

# Instancia o motor do YOLO carregando os pesos e hardware definidos no config.py
yolo_service = YoloService()

@router.post("/detect", summary="Realiza a detecção de objetos em uma imagem")
async def detect_objects(file: UploadFile = File(...)):
    """
    Endpoint HTTP POST que aceita o upload de um arquivo binário de imagem.
    Retorna um objeto JSON estruturado contendo a classe, confiança e as caixas delimitadoras.
    """
    # 1. Validação básica de extensão do arquivo para segurança do microsserviço
    extensoes_suportadas = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if not file.filename.lower().endswith(extensoes_suportadas):
        raise HTTPException(
            status_code=400, 
            detail=f"Formato de arquivo inválido. Extensões suportadas: {', '.join(extensoes_suportadas)}"
        )
    
    try:
        # 2. Lê os bytes brutos do arquivo carregado na requisição assíncrona
        imagem_bytes = await file.read()
        
        # 3. Dispara o pipeline de inferência (Híbrido AMD / NVIDIA / CPU)
        resultado = yolo_service.detectar_objetos(imagem_bytes)
        
        # 4. Se o processamento matricial falhar por imagem corrompida, levanta erro técnico
        if resultado.get("status") == "error":
            raise HTTPException(status_code=422, detail=resultado.get("message"))
            
        # 5. Devolve o payload estruturado exatamente igual ao solicitado no edital
        return resultado
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erro interno ao processar a inferência no microsserviço: {str(e)}"
        )
