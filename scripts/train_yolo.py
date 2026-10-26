import argparse
import os
import torch
from ultralytics import YOLO

def detectar_dispositivo_treino():
    """
    Detecta e configura o hardware para o treinamento do YOLOv8s.
    Força CPU para AMD devido à incompatibilidade do DirectML com loops de treino da Ultralytics.
    """
    if torch.cuda.is_available():
        print("🚀 GPU NVIDIA Detectada! O treinamento rodará via CUDA.")
        return 0  # Índice da GPU para o PyTorch/YOLO
        
    try:
        import torch_directml
        # Detectamos se é AMD apenas para avisar o usuário sobre o desvio seguro
        print("⚠️ GPU AMD Detectada! Devido a limitações do DirectML com a Ultralytics, o treino será desviado para a CPU para evitar travamentos.")
        return "cpu"
    except Exception:
        print("💻 Rodando o treino no CPU.")
        return "cpu"

def treinar_modelo():
    parser = argparse.ArgumentParser(description="Treinamento Adaptativo do YOLOv8s com Suporte AMD/NVIDIA")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        choices=["placas", "ingredientes"], 
        help="Escolha o cenário de treino: 'placas' ou 'ingredientes'"
    )
    args = parser.parse_args()

    yaml_config = f"../data_{args.dataset}.yaml"
    if not os.path.exists(yaml_config):
        yaml_config = f"data_{args.dataset}.yaml"
        if not os.path.exists(yaml_config):
            print(f"❌ Erro: O arquivo de configuração {yaml_config} não foi localizado.")
            return

    dispositivo = detectar_dispositivo_treino()

    print(f"\n==============================================================")
    print(f"🎯 Dataset Ativo: {args.dataset.upper()}")
    print(f"📄 Arquivo YAML: {yaml_config}")
    print(f"==============================================================\n")

    # Carrega o modelo base Small nativo (.pt) para iniciar o Fine-Tuning
    modelo_base = "../models/yolov8s.pt" if os.path.exists("../models/yolov8s.pt") else "yolov8s.pt"
    model = YOLO(modelo_base)

    # Dispara o treino com tratamento de workers para evitar crashes no Windows
    model.train(
        data=yaml_config,
        epochs=50,
        imgsz=640,
        batch=16,
        device=dispositivo,
        project="../models/treinamentos",
        name=f"yolo_{args.dataset}",
        workers=0 if os.name == 'nt' else 2  # 0 evita gargalo de memória compartilhada no Windows
    )

    print(f"\n🔄 Exportando o modelo treinado para o formato ONNX universal...")
    # Carrega o melhor peso gerado em formato nativo do PyTorch
    caminho_melhor_peso = f"../models/treinamentos/yolo_{args.dataset}/weights/best.pt"
    model_treinado = YOLO(caminho_melhor_peso)
    
    # Exporta para ONNX (O framework gera o arquivo best.onnx na mesma pasta)
    model_treinado.export(format="onnx", imgsz=640)

    print(f"\n==============================================================")
    print(f"✅ CONCLUÍDO COM SUCESSO!")
    print(f"💾 Modelo PyTorch: ../models/treinamentos/yolo_{args.dataset}/weights/best.pt")
    print(f"💾 Modelo ONNX:    ../models/treinamentos/yolo_{args.dataset}/weights/best.onnx")
    print(f"==============================================================\n")

if __name__ == "__main__":
    treinar_modelo()
