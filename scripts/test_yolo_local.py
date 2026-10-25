import os
import cv2
import torch
import onnx
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

def detectar_dispositivo():
    """Detecta de forma adaptativa o hardware local."""
    if torch.cuda.is_available():
        print("🚀 GPU NVIDIA Detectada! Inferência via CUDA.")
        return "cuda:0", False
    try:
        import torch_directml
        dml_device = torch_directml.device()
        print(f"🚀 GPU AMD/Intel Detectada! Inferência via DirectML ({dml_device}).")
        return dml_device, True
    except Exception:
        print("💻 Nenhuma GPU configurada. Rodando via CPU.")
        return "cpu", False

def processar_teste_local():
    dispositivo, eh_amd = detectar_dispositivo()

    # 1. Definição do cenário e do modelo ativo para o teste rápido
    dataset_ativo = "placas"
    
    # Você pode alternar entre o baseline padrão (.onnx) ou o seu modelo treinado
    # caminho_modelo = "../models/yolov8s.onnx"
    caminho_modelo = f"../models/treinamentos/yolo_{dataset_ativo}/weights/best.onnx"

    if not os.path.exists(caminho_modelo):
        print(f"❌ Erro: O arquivo de pesos '{caminho_modelo}' não foi localizado.")
        return

    # 2. Correção de metadados ONNX (Bypass de compatibilidade universal)
    try:
        model_proto = onnx.load(caminho_modelo)
        if model_proto.ir_version > 13:
            print(f"🔄 Convertendo metadados do ONNX de IR {model_proto.ir_version} para IR 13...")
            model_proto.ir_version = 13
            onnx.save(model_proto, caminho_modelo)
    except Exception as e:
        print(f"Nota: Metadados ONNX inalterados. Razão: {e}")

    # 3. Carregamento do modelo e extração das classes
    nomes_classes = {}
    if eh_amd:
        ort_session = ort.InferenceSession(caminho_modelo, providers=['CPUExecutionProvider'])
        meta = ort_session.get_modelmeta().custom_metadata_map
        if 'names' in meta:
            nomes_classes = eval(meta['names'])
    else:
        model = YOLO(caminho_modelo)
        model.to(dispositivo)
        nomes_classes = model.names

    # 4. Configuração das imagens de entrada e diretório de saída
    os.makedirs("../resultados_teste", exist_ok=True)
    
    # Coloque aqui os caminhos das imagens que você quer validar rapidamente por terminal
    imagens_teste = [
        "../tests/foto_rua_carros.jpg",
        "../tests/foto_gato_teste.jpg"
    ]

    print(f"\n🎯 Iniciando Varredura de Testes para o cenário: {dataset_ativo.upper()}")

    for img_path in imagens_teste:
        if not os.path.exists(img_path):
            print(f"⚠️ Arquivo {img_path} não encontrado. Pulando...")
            continue

        print(f"--- Processando: {os.path.basename(img_path)} ---")
        img_bgr = cv2.imread(img_path)
        h_orig, w_orig = img_bgr.shape[:2]
        caixas_detectadas = []

        # 5. Inferência condicionada ao ecossistema de hardware
        if eh_amd:
            img_resized = cv2.resize(img_bgr, (640, 640))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(img_rgb.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0
            
            input_name = ort_session.get_inputs()[0].name
            outputs = ort_session.run(None, {input_name: input_data})
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
                    caixas_detectadas.append((x_min, y_min, x_max, y_max, conf, cls_id))
        else:
            with torch.inference_mode():
                resultados = model(img_path, imgsz=640, conf=0.25, device=dispositivo)
            for result in resultados:
                for box in result.boxes:
                    conf = float(box.conf)
                    if conf > 0.4:
                        cls_id = int(box.cls)
                        x_min, y_min, x_max, y_max = box.xyxy.tolist()[0]
                        caixas_detectadas.append((int(x_min), int(y_min), int(x_max), int(y_max), conf, cls_id))

        # 6. Desenho das predições e gravação do arquivo físico local
        for x_min, y_min, x_max, y_max, conf, cls_id in caixas_detectadas:
            nome_classe = nomes_classes.get(cls_id, f"Classe_{cls_id}")
            print(f"   ↳ [DETECTADO] {nome_classe} ({conf:.2f}) -> Box: [{x_min}, {y_min}, {x_max}, {y_max}]")
            
            cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(img_bgr, f"{nome_classe} {conf:.2f}", (x_min, max(y_min - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        nome_saida = f"resultado_{os.path.basename(img_path)}"
        caminho_salvar = os.path.join("../resultados_teste", nome_saida)
        cv2.imwrite(caminho_salvar, img_bgr)
        print(f"💾 Foto processada salva em: {caminho_salvar}\n")

if __name__ == "__main__":
    processar_teste_local()
