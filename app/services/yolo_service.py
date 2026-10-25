import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
from app.config import MODEL_PATH, EH_AMD, DISPOSITIVO_ATIVO

class YoloService:
    def __init__(self):
        self.nomes_classes = {}
        self.eh_amd = EH_AMD

        if self.eh_amd:
            # Fluxo exclusivo AMD: Inicializa sessão pura do ONNX Runtime em CPU/DirectML Provider
            self.ort_session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
            meta = self.ort_session.get_modelmeta().custom_metadata_map
            if 'names' in meta:
                self.nomes_classes = eval(meta['names'])
            else:
                raise RuntimeError("Falha ao ler o dicionário de classes do arquivo ONNX.")
        else:
            # Fluxo Padrão: Carrega o modelo via SDK Ultralytics e aponta o dispositivo correto
            self.model = YOLO(MODEL_PATH)
            self.model.to(DISPOSITIVO_ATIVO)
            self.nomes_classes = self.model.names

    def detectar_objetos(self, imagem_bytes: bytes) -> dict:
        # Convertendo o payload binário HTTP recebido pelo FastAPI em matriz OpenCV
        nparr = np.frombuffer(imagem_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return {"status": "error", "message": "Imagem corrompida ou inválida."}

        h_orig, w_orig = img_bgr.shape[:2]
        detections = []

        if self.eh_amd:
            # --- INTEGRAÇÃO DO SEU PIPELINE DE EXECUÇÃO EXCLUSIVO AMD ---
            img_resized = cv2.resize(img_bgr, (640, 640))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Formatação BCHW (1, 3, 640, 640) e normalização matemática
            input_data = np.expand_dims(img_rgb.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0
            input_name = self.ort_session.get_inputs()[0].name
            
            outputs = self.ort_session.run(None, {input_name: input_data})
            predictions = np.squeeze(outputs[0]).T
            
            for pred in predictions:
                x_center, y_center, width, height = pred[:4]
                scores = pred[4:]
                cls_id = int(np.argmax(scores))
                conf = float(scores[cls_id])
                
                if conf > 0.4:  # Limiar de corte definido no seu Teste 1
                    # Regra de três para remapear do tamanho (640,640) de volta aos pixels da imagem original
                    x_min = int((x_center - width / 2) * (w_orig / 640.0))
                    y_min = int((y_center - height / 2) * (h_orig / 640.0))
                    x_max = int((x_center + width / 2) * (w_orig / 640.0))
                    y_max = int((y_center + height / 2) * (h_orig / 640.0))
                    
                    detections.append({
                        "class_name": self.nomes_classes.get(cls_id, f"Classe_{cls_id}"),
                        "confidence": round(conf, 2),
                        "box": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}
                    })
        else:
            # --- INTEGRAÇÃO DO PIPELINE PADRÃO NVIDIA CUDA / CPU ---
            import torch
            with torch.inference_mode():
                resultados = self.model(img_bgr, imgsz=640, conf=0.25, device=DISPOSITIVO_ATIVO)
                
            for result in resultados:
                for box in result.boxes:
                    conf = float(box.conf)
                    if conf > 0.4:
                        cls_id = int(box.cls)
                        x_min, y_min, x_max, y_max = box.xyxy.tolist()[0]
                        
                        detections.append({
                            "class_name": self.nomes_classes.get(cls_id, f"Classe_{cls_id}"),
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
            "detections": detections
        }
