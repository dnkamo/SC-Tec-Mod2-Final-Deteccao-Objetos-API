import os
import torch
import onnx
import urllib.request

# Caminhos base da árvore de diretórios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Define a extensão prioritária do arquivo (.onnx para AMD / .pt para NVIDIA-CPU)
EXTENSAO_MODELO = ".onnx" 

BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, f"yolov8s{EXTENSAO_MODELO}")

# === DOWNLOAD AUTOMÁTICO DO MODELO PADRÃO (FALLBACK BACKUP) ===
if EXTENSAO_MODELO == ".onnx" and not os.path.exists(BASELINE_MODEL_PATH):
    print("📥 [CONFIG] Modelo padrão não encontrado. Baixando versão oficial convertida...")
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        url_onnx = "https://github.com"
        urllib.request.urlretrieve(url_onnx, BASELINE_MODEL_PATH)
        print("✅ [CONFIG] Download do modelo padrão concluído com sucesso!")
    except Exception as e:
        print(f"❌ [CONFIG] Falha ao baixar o modelo padrão automaticamente: {e}")

# === DETECÇÃO DINÂMICA DE HARDWARE (AMD DirectML vs NVIDIA CUDA vs CPU) ===
def detectar_hardware_api():
    if torch.cuda.is_available():
        return "cuda:0", False
    try:
        import torch_directml
        dml_device = torch_directml.device()
        return dml_device, True
    except Exception:
        return "cpu", False

DISPOSITIVO_ATIVO, EH_AMD = detectar_hardware_api()

# === CORREÇÃO AUTOMÁTICA DE METADADOS DO INTERMEDIATE REPRESENTATION (IR) ONNX ===
if BASELINE_MODEL_PATH.endswith(".onnx") and os.path.exists(BASELINE_MODEL_PATH):
    try:
        model_proto = onnx.load(BASELINE_MODEL_PATH)
        if model_proto.ir_version > 13:
            model_proto.ir_version = 13
            onnx.save(model_proto, BASELINE_MODEL_PATH)
            print(f"🔄 [CONFIG] Downgrade automático do ONNX para IR 13 aplicado em: {BASELINE_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ [CONFIG] Não foi necessário ajustar os metadados do ONNX base. Motivo: {e}")
