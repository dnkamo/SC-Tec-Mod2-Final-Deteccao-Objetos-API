import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io

# Configuração da página web para modo amplo (melhor aproveitamento de tela no vídeo)
st.set_page_config(page_title="VisionGuard AI - Interface de Detecção", layout="wide")

st.title("👁️ Sistema de Visão Computacional - YOLOv8s + FastAPI")
st.markdown("---")

# URL estável do microsserviço FastAPI configurado na Etapa 3
API_URL = "http://127.0.0.1:8000/detect"

# ==============================================================================
# PAINEL LATERAL DE CONTROLES (CONFIGURAÇÕES DO PRODUTO)
# ==============================================================================
st.sidebar.header("⚙️ Painel de Controle")

# Caixa de seleção com os nomes exatos esperados pelo parâmetro Literal da nossa API
cenario_opcao = st.sidebar.selectbox(
    "Selecione a Inteligência Ativa:",
    options=["placas", "ingredientes", "padrao"],
    format_func=lambda x: {
        "placas": "🚦 Sinais de Trânsito (Custom)",
        "ingredientes": "🥦 Ingredientes/Vegetais (Custom)",
        "padrao": "📦 Modelo Baseline Padrão (COCO)"
    }[x]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Engenharia de Software & MLOps:**\n\n"
    "Este front-end opera de forma 100% isolada, consumindo o microsserviço "
    "FastAPI de maneira assíncrona e desacoplada."
)

# ==============================================================================
# ÁREA PRINCIPAL DA APLICAÇÃO (UPLOADS E INFERÊNCIAS)
# ==============================================================================
uploaded_file = st.file_uploader(
    "Arraste ou escolha uma imagem para realizar a inferência no modelo ativo...", 
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    # Captura e clona os bytes originais imediatamente em memória para evitar buffers vazios
    bytes_data = uploaded_file.getvalue()
    
    # Carrega a imagem para exibição local no Streamlit usando os bytes isolados
    image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    
    # Divide a tela em duas colunas simétricas para uma exibição didática no vídeo
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Imagem Original do Cliente")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("🚀 Resposta Visual com Bounding Boxes")
        
        # Botão interativo para disparar a requisição de inferência
        if st.button("Executar Detecção de Objetos", type="primary"):
            with st.spinner(f"Despachando imagem para o cenário '{cenario_opcao}' no backend..."):
                try:
                    # Estrutura o arquivo a partir do dump de bytes isolado em memória (Garante envio íntegro)
                    files = {"file": (uploaded_file.name, bytes_data, uploaded_file.type)}
                    
                    # Constrói a URL injetando dinamicamente o parâmetro de consulta de cenário
                    url_completa = f"{API_URL}?cenario={cenario_opcao}"
                    
                    # Dispara a chamada HTTP POST adicionando timeout de segurança contra travamentos de GPU
                    response = requests.post(
                        url_completa, 
                        files=files, 
                        timeout=30, 
                        proxies={"http": None, "https": None}
                    )
                    
                    if response.status_code == 200:
                        resultado_json = response.json()
                        detections = resultado_json.get("detections", [])
                        
                        if not detections:
                            st.warning(f"Nenhum objeto do escopo de '{cenario_opcao}' foi localizado acima do limiar configurado.")
                            st.image(image, use_column_width=True)
                        else:
                            # Converte em array NumPy e cria uma cópia explícita gravável na memória
                            img_np = np.array(image).copy()
                            
                            # Converte de RGB para BGR antes do desenho para manter a integridade cromática do OpenCV
                            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                            # Desenha os componentes geométricos recebidos estritamente do JSON do backend
                            for det in detections:
                                classe = det["class_name"]
                                conf = det["confidence"]
                                box = det["box"]
                                
                                # Coordenadas em pixels reais convertidas para inteiros
                                x_min, y_min = int(box["x_min"]), int(box["y_min"])
                                x_max, y_max = int(box["x_max"]), int(box["y_max"])
                                
                                # Desenha o retângulo delimitador verde (BBox)
                                cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                                
                                # Estrutura a etiqueta de texto (Nome da Classe + Confiança)
                                etiqueta = f"{classe} {conf:.2f}"
                                
                                cv2.putText(
                                    img_np, 
                                    etiqueta, 
                                    (x_min, max(y_min - 10, 0)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.6, 
                                    (0, 255, 0), 
                                    2
                                )
                            
                            # Retorna para RGB para o Streamlit renderizar as cores corretamente na tela
                            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                            
                            # Renderiza a imagem final processada com as caixas
                            st.image(img_np, use_column_width=True)
                            st.success(f"Inferência concluída! Processado via IA: {resultado_json.get('cenario_executado')}")
                            
                            # Exibe o payload estruturado logo abaixo (Exigência do edital)
                            st.markdown("### 📄 Payload JSON Estruturado:")
                            st.json(resultado_json)
                            
                    else:
                        st.error(f"O Microsserviço Backend retornou um erro técnico: Código {response.status_code}")
                        st.code(response.text)
                        
                except requests.exceptions.Timeout:
                    st.error("Tempo limite de requisição esgotado! O backend demorou muito para responder (verifique se a GPU/CPU travou).")
                except requests.exceptions.ConnectionError:
                    st.error(
                        "Falha de conexão com a API Backend! Certifique-se de que o servidor "
                        "Uvicorn está rodando ativamente na porta 8000 antes de disparar o Streamlit."
                    )
