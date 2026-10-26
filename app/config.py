import os
import torch
import onnx

# Define qual cenário a API vai rodar: "placas" ou "ingredientes
# "
DATASET_ATIVO = "placas"

# Caminhos base da árvore de diretórios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Define a extensão prioritária do arquivo com base no seu novo ambiente (.onnx para AMD / .pt para NVIDIA-CPU)
# Você pode mudar para ".pt" se for usar o modelo nativo em ambiente CUDA
EXTENSAO_MODELO = ".onnx" 

# Mapeamento do caminho do modelo ativo
CUSTOM_MODEL_PATH = os.path.join(MODELS_DIR, "treinamentos", f"yolo_{DATASET_ATIVO}", "weights", f"best{EXTENSAO_MODELO}")
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, f"yolov8s{EXTENSAO_MODELO}")

if os.path.exists(CUSTOM_MODEL_PATH):
    MODEL_PATH = CUSTOM_MODEL_PATH
else:
    MODEL_PATH = BASELINE_MODEL_PATH

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
if MODEL_PATH.endswith(".onnx") and os.path.exists(MODEL_PATH):
    try:
        model_proto = onnx.load(MODEL_PATH)
        if model_proto.ir_version > 13:
            model_proto.ir_version = 13
            onnx.save(model_proto, MODEL_PATH)
            print(f"🔄 [CONFIG] Downgrade automático do ONNX para IR 13 aplicado em: {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ [CONFIG] Não foi necessário ajustar os metadados do ONNX. Motivo: {e}")
