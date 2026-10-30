import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io
import json

# Configuração da página web para modo amplo
st.set_page_config(page_title="VisionGuard AI - Dashboard de Inteligência", layout="wide")

# ==============================================================================
# ESTILIZAÇÃO VISUAL MODERNIZADA (CUSTOM CSS)
# ==============================================================================
st.markdown("""
    <style>
        .metric-card {
            background-color: #1E293B;
            padding: 18px;
            border-radius: 12px;
            border-left: 5px solid #10B981;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 15px;
        }
        section[data-testid="stFileUploader"] {
            border: 2px dashed #4B5563;
            border-radius: 14px;
            padding: 25px;
            background-color: #0F172A;
        }
        .main-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(90deg, #10B981, #3B82F6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            color: #9CA3AF;
            font-size: 1.1rem;
            font-style: italic;
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">👁️ VisionGuard AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Plataforma Avançada de Visão Computacional - YOLOv8s + FastAPI</p>', unsafe_allow_html=True)
st.markdown("---")

API_URL = "http://127.0.0.1:8000/detect"

def limpar_estado_deteccao():
    st.session_state.resultado_json = None
    st.session_state.img_renderizavel = None
    st.session_state.image_bytes_processada = None

# Inicialização segura na memória de sessão
if "resultado_json" not in st.session_state:
    limpar_estado_deteccao()

# Painel Lateral de Controles
st.sidebar.header("⚙️ Painel de Controle")
cenario_opcao = st.sidebar.selectbox(
    "Selecione a Inteligência Ativa:",
    options=["placas", "ingredientes", "padrao"],
    format_func=lambda x: {
        "placas": "🚦 Sinais de Trânsito (Custom)",
        "ingredientes": "🥦 Ingredientes/Vegetais (Custom)",
        "padrao": "📦 Modelo Baseline Padrão (COCO)"
    }[x],
    on_change=limpar_estado_deteccao
)

st.sidebar.markdown("---")
st.sidebar.info("**Engenharia de Software & MLOps:** Front-end desacoplado e assíncrono.")

# Área Principal: Upload de imagem associado ao callback de limpeza automática
uploaded_file = st.file_uploader(
    "Arraste ou escolha uma imagem para realizar a inferência no modelo ativo...", 
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    key="uploader_imagem_principal",
    on_change=limpar_estado_deteccao
)

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Imagem Original do Cliente")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("🚀 Resposta Visual e Análise")
        
        if st.button("Executar Detecção de Objetos", type="primary"):
            with st.spinner(f"Processando no cenário '{cenario_opcao}'..."):
                try:
                    files = {"file": (uploaded_file.name, bytes_data, uploaded_file.type)}
                    url_completa = f"{API_URL}?cenario={cenario_opcao}"
                    
                    response = requests.post(url_completa, files=files, timeout=30, proxies={"http": None, "https": None})
                    
                    if response.status_code == 200:
                        resultado_json = response.json()
                        detections = resultado_json.get("detections", [])
                        
                        if detections:
                            CORES_CENARIOS = {"placas": (0, 0, 255), "ingredientes": (0, 255, 0), "padrao": (255, 165, 0)}
                            cor_atual = CORES_CENARIOS.get(cenario_opcao, (0, 255, 0))

                            img_np = np.array(image).copy()
                            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                            for det in detections:
                                classe = det["class_name"]
                                conf = det["confidence"]
                                box = det["box"]
                                x_min, y_min = int(box["x_min"]), int(box["y_min"])
                                x_max, y_max = int(box["x_max"]), int(box["y_max"])
                                
                                cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), cor_atual, 3)
                                etiqueta = f"{classe} {conf:.2f}"
                                cv2.putText(img_np, etiqueta, (x_min, max(y_min - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_atual, 2)
                            
                            img_np_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                            
                            # Gera os bytes da imagem final de forma isolada para evitar corrupção no download
                            img_pil_download = Image.fromarray(img_np_rgb)
                            buf = io.BytesIO()
                            img_pil_download.save(buf, format="JPEG")
                            
                            st.session_state.resultado_json = resultado_json
                            st.session_state.img_renderizavel = img_np_rgb
                            st.session_state.image_bytes_processada = buf.getvalue()
                        else:
                            st.session_state.resultado_json = {"status": "success", "message": "Nenhum objeto detectado.", "detections": []}
                            st.session_state.img_renderizavel = np.array(image)
                            st.session_state.image_bytes_processada = bytes_data

                        # Força o Streamlit a recarregar a página e exibir os dados imediatamente
                        st.rerun()
                        
                    else:
                        st.error(f"Erro no Backend: Código {response.status_code}")
                        st.code(response.text)
                except Exception as e:
                    st.error(f"Falha de conexão com a API Backend: {e}")

        # Renderização do estado salvo (Segurança total contra travamentos pós-download)
        if st.session_state.resultado_json is not None:
            detections = st.session_state.resultado_json.get("detections", [])
            
            if detections:
                total_detectado = len(detections)
                contagem_classes = {}
                for det in detections:
                    c_name = det["class_name"]
                    contagem_classes[c_name] = contagem_classes.get(c_name, 0) + 1
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f'<div class="metric-card"><b>Total Encontrado</b><br><span style="font-size:24px; color:#10B981; font-weight:bold;">{total_detectado} objetos</span></div>', unsafe_allow_html=True)
                with m2:
                    hardware_usado = st.session_state.resultado_json.get('cenario_executado', cenario_opcao)
                    st.markdown(f'<div class="metric-card" style="border-left-color:#3B82F6;"><b>Modelo / Contexto</b><br><span style="font-size:16px; font-weight:500;">{hardware_usado}</span></div>', unsafe_allow_html=True)
                with m3:
                    maior_classe = max(contagem_classes, key=contagem_classes.get) if contagem_classes else "Nenhum"
                    st.markdown(f'<div class="metric-card" style="border-left-color:#F59E0B;"><b>Predominante</b><br><span style="font-size:18px; font-weight:600; color:#F59E0B;">{maior_classe}</span></div>', unsafe_allow_html=True)

                with st.expander("📊 Ver distribuição detalhada por classe"):
                    st.bar_chart(contagem_classes)
            else:
                st.info("A inferência rodou, mas nenhum objeto do escopo selecionado foi localizado acima do limiar configurado.")

            if st.session_state.img_renderizavel is not None:
                st.image(st.session_state.img_renderizavel, use_column_width=True)

            # Painel Estável de Downloads
            st.markdown("### 📥 Exportar Resultados")
            col_down1, col_down2 = st.columns(2)
            
            with col_down1:
                json_string = json.dumps(st.session_state.resultado_json, indent=4).encode('utf-8')
                st.download_button(
                    label="📄 Baixar Relatório JSON",
                    data=json_string,
                    file_name=f"relatorio_visao_{cenario_opcao}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col_down2:
                if st.session_state.image_bytes_processada is not None:
                    st.download_button(
                        label="🖼️ Salvar Imagem Processada",
                        data=st.session_state.image_bytes_processada,
                        file_name=f"resultado_deteccao_{cenario_opcao}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )

            st.markdown("---")
            st.markdown("### 📄 Payload JSON Estruturado:")
            st.json(st.session_state.resultado_json)
else:
    limpar_estado_deteccao()
