import os
import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
import onnx
from app.config import MODELS_DIR, BASELINE_MODEL_PATH, EXTENSAO_MODELO, EH_AMD, DISPOSITIVO_ATIVO

class YoloService:
    def __init__(self):
        self.eh_amd = EH_AMD
        # Cache em memória para evitar recarregar o arquivo do disco a cada requisição
        self.modelos_carregados = {}

    def _obter_caminho_modelo(self, cenario: str) -> tuple[str, str]:
        """Retorna uma tupla (caminho_final, cenario_real) aplicando o fallback de segurança."""
        if cenario == "padrao":
            return BASELINE_MODEL_PATH, "padrao"
            
        custom_path = os.path.join(MODELS_DIR, "treinamentos", f"yolo_{cenario}", "weights", f"best{EXTENSAO_MODELO}")
        
        # Se o modelo customizado não existir no disco, usa o padrão como segurança
        if os.path.exists(custom_path):
            return custom_path, cenario
        
        print(f"⚠️ [AVISO] Modelo para o cenário '{cenario}' não foi encontrado. Usando 'padrao' como fallback.")
        return BASELINE_MODEL_PATH, "padrao (fallback)"

    def _garantir_metadados_onnx(self, caminho_modelo: str):
        """Aplica o downgrade de IR se um novo modelo ONNX customizado for carregado."""
        if caminho_modelo.endswith(".onnx") and os.path.exists(caminho_modelo):
            try:
                model_proto = onnx.load(caminho_modelo)
                if model_proto.ir_version > 13:
                    model_proto.ir_version = 13
                    onnx.save(model_proto, caminho_modelo)
                    print(f"🔄 [SERVICE] Downgrade de IR 13 aplicado no modelo customizado: {caminho_modelo}")
            except Exception:
                pass

    def detectar_objetos(self, imagem_bytes: bytes, cenario: str = "padrao") -> dict:
        nparr = np.frombuffer(imagem_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return {"status": "error", "message": "Imagem corrompida ou inválida."}

        h_orig, w_orig = img_bgr.shape[:2]
        detections = []

        # Resolve o caminho real aplicando as regras de validação de arquivo
        caminho_final, cenario_real = self._obter_caminho_modelo(cenario)

        # Inicialização sob demanda com gerenciamento de Cache
        if caminho_final not in self.modelos_carregados:
            self._garantir_metadados_onnx(caminho_final)
            
            if self.eh_amd:
                session = ort.InferenceSession(caminho_final, providers=['CPUExecutionProvider'])
                meta = session.get_modelmeta().custom_metadata_map
                classes = eval(meta['names']) if 'names' in meta else {}
                self.modelos_carregados[caminho_final] = {"session": session, "classes": classes}
            else:
                modelo_yolo = YOLO(caminho_final)
                modelo_yolo.to(DISPOSITIVO_ATIVO)
                self.modelos_carregados[caminho_final] = {"model": modelo_yolo, "classes": modelo_yolo.names}

        instancia = self.modelos_carregados[caminho_final]
        nomes_classes = instancia["classes"]

        if self.eh_amd:
            # --- PIPELINE EXCLUSIVO AMD (Fiel ao original do usuário) ---
            img_resized = cv2.resize(img_bgr, (640, 640))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(img_rgb.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0
            
            # Correção aqui: Adicionado o índice [0] corrigindo o bug mapeado
            input_name = instancia["session"].get_inputs()[0].name
            
            outputs = instancia["session"].run(None, {input_name: input_data})
            predictions = np.squeeze(outputs[0]).T
            
            for pred in predictions:
                x_center, y_center, width, height = pred[:4]
                scores = pred[4:]
                cls_id = int(np.argmax(scores))
                conf = float(scores[cls_id])
                
                if conf > 0.4:
                    x_min = int((x_center - width / 2) * (w_orig / 640.0))
                    y_min = int((y_center - height / 2) * (h_orig / 640.0))
                    x_max = int((x_center + width / 2) * (w_orig / 640.0))
                    y_max = int((y_center + height / 2) * (h_orig / 640.0))
                    
                    detections.append({
                        "class_name": nomes_classes.get(cls_id, f"Classe_{cls_id}"),
                        "confidence": round(conf, 2),
                        "box": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}
                    })
        else:
            # --- PIPELINE PADRÃO NVIDIA CUDA / CPU (Fiel ao original do usuário) ---
            import torch
            with torch.inference_mode():
                resultados = instancia["model"](img_bgr, imgsz=640, conf=0.25, device=DISPOSITIVO_ATIVO)
                
            for result in resultados:
                for box in result.boxes:
                    conf = float(box.conf)
                    if conf > 0.4:
                        cls_id = int(box.cls)
                        # Correção aqui: Adicionado o índice [0] corrigindo o bug mapeado
                        x_min, y_min, x_max, y_max = box.xyxy.tolist()[0]
                        
                        detections.append({
                            "class_name": nomes_classes.get(cls_id, f"Classe_{cls_id}"),
                            "confidence": round(conf, 2),
                            "box": {
                                "x_min": int(x_min),
                                "y_min": int(y_min),
                                "x_max": int(x_max),
                                "y_max": int(y_max)
                            }
                        })

        return {
            "status": "success",
            "cenario_solicitado": cenario,
            "cenario_executado": cenario_real,
            "detections": detections
        }
