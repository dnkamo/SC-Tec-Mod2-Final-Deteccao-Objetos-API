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

    # ==============================================================================
    # CONTROLE CENTRAL: ALTERNE APENAS ESTA VARIÁVEL ("placas" ou "ingredientes")
    # ==============================================================================
    DATASET_ATIVO = "placas" 
    
    # Descobre o diretório onde este arquivo de script .py está salvo
    DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Mapeamento Dinâmico e Absoluto de Modelos e Imagens
    if DATASET_ATIVO in ["placas", "ingredientes"]:
        caminho_modelo = os.path.abspath(os.path.join(DIRETORIO_ATUAL, "..", "models", "treinamentos", f"yolo_{DATASET_ATIVO}", "weights", "best.onnx"))
        pasta_imagens_teste = os.path.abspath(os.path.join(DIRETORIO_ATUAL, "..", "tests", DATASET_ATIVO))
    else:
        print(f"Erro: Cenário '{DATASET_ATIVO}' desconhecido.")
        return

    if not os.path.exists(caminho_modelo):
        print(f"Erro: O arquivo de pesos '{caminho_modelo}' não foi localizado.")
        return

    # 2. Correção de metadados ONNX (Bypass de compatibilidade universal IR 13)
    try:
        model_proto = onnx.load(caminho_modelo)
        if model_proto.ir_version > 13:
            print(f"Convertendo metadados do ONNX de IR {model_proto.ir_version} para IR 13...")
            model_proto.ir_version = 13
            onnx.save(model_proto, caminho_modelo)
    except Exception as e:
        print(f"Nota: Metadados ONNX inalterados. Razão: {e}")

    # 3. Carregamento do modelo e extração das classes de forma dinâmica
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

    # 4. Varredura Automática de Imagens na pasta do Dataset Ativo
    if not os.path.exists(pasta_imagens_teste):
        print(f"Pasta de testes não localizada em: {pasta_imagens_teste}. Criando pasta vazia...")
        os.makedirs(pasta_imagens_teste, exist_ok=True)
        return

    # O filtro abaixo lista os arquivos REAIS da pasta, independente do nome complexo do Roboflow
    extensoes_validas = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    imagens_teste = [
        os.path.join(pasta_imagens_teste, arquivo) 
        for arquivo in os.listdir(pasta_imagens_teste) 
        if arquivo.lower().endswith(extensoes_validas)
    ]

    if not imagens_teste:
        print(f"Nenhuma imagem encontrada dentro da pasta correspondente: '{pasta_imagens_teste}'")
        return

    # Cria diretório de saída unificado na raiz
    pasta_resultados = os.path.join("..", "resultados_teste", DATASET_ATIVO)
    os.makedirs(pasta_resultados, exist_ok=True)

    print(f"\n======================================================================")
    print(f"SCENARIO ATIVO: {DATASET_ATIVO.upper()}")
    print(f"Total de imagens Roboflow detectadas para teste: {len(imagens_teste)}")
    print(f"Salvando resultados em: {pasta_resultados}")
    print(f"======================================================================\n")

    for img_path in imagens_teste:
        # Pega o nome complexo original para salvar o resultado com o mesmo nome
        nome_arquivo_original = os.path.basename(img_path)
        print(f"--- Processando arquivo: {nome_arquivo_original} ---")
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Erro ao abrir a imagem: {img_path}")
            continue
            
        h_orig, w_orig = img_bgr.shape[:2]
        caixas_detectadas = []

        # 5. Inferência condicionada ao ecossistema de hardware
        caixas_candidatas = []
        confidencias_candidatas = []
        classes_candidatas = []

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
                
                if conf > 0.25:  # Limiar inicial de confiança
                    # Converte para formato do OpenCV: [x_min, y_min, largura, altura]
                    w_box = int(width * (w_orig / 640.0))
                    h_box = int(height * (h_orig / 640.0))
                    x_min = int((x_center - width / 2) * (w_orig / 640.0))
                    y_min = int((y_center - height / 2) * (h_orig / 640.0))
                    
                    caixas_candidatas.append([x_min, y_min, w_box, h_box])
                    confidencias_candidatas.append(conf)
                    classes_candidatas.append(cls_id)
        else:
            with torch.inference_mode():
                # O parâmetro iou=0.45 controla a supressão nativa da Ultralytics
                resultados = model(img_path, imgsz=640, conf=0.25, iou=0.45, device=dispositivo)
            for result in resultados:
                for box in result.boxes:
                    conf = float(box.conf)
                    if conf > 0.25:
                        cls_id = int(box.cls)
                        x_min, y_min, x_max, y_max = box.xyxy.tolist()[0]
                        w_box = int(x_max - x_min)
                        h_box = int(y_max - y_min)
                        
                        caixas_candidatas.append([int(x_min), int(y_min), w_box, h_box])
                        confidencias_candidatas.append(conf)
                        classes_candidatas.append(cls_id)

        # --- APLICAÇÃO DO NMS (SUPRESSÃO NÃO-MÁXIMA) ---
        # score_threshold: ignora caixas abaixo dessa confiança
        # nms_threshold: caixas que se sobrepõem mais que 45% (IoU) serão suprimidas
        indices_finais = cv2.dnn.NMSBoxes(
            bboxes=caixas_candidatas, 
            scores=confidencias_candidatas, 
            score_threshold=0.40, 
            nms_threshold=0.45
        )

        # 6. Desenho das predições filtradas pelo NMS
        if len(indices_finais) > 0:
            # Garante compatibilidade de indexação entre diferentes versões do OpenCV
            for i in indices_finais.flatten():
                x_min, y_min, w_box, h_box = caixas_candidatas[i]
                x_max, y_max = x_min + w_box, y_min + h_box
                conf = confidencias_candidatas[i]
                cls_id = classes_candidatas[i]
                
                nome_classe = nomes_classes.get(cls_id, f"Classe_{cls_id}")
                print(f"   ↳ [DETECTADO] {nome_classe} ({conf:.2f}) -> Box: [{x_min}, {y_min}, {x_max}, {y_max}]")
                
                cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(img_bgr, f"{nome_classe} {conf:.2f}", (x_min, max(y_min - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        nome_saida = f"resultado_{os.path.basename(img_path)}"
        caminho_salvar = os.path.join(pasta_resultados, nome_saida)
        cv2.imwrite(caminho_salvar, img_bgr)
        print(f"Foto salva em: {caminho_salvar}\n")

if __name__ == "__main__":
    processar_teste_local()
