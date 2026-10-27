import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import torch

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

    # 1. MAPEAMENTO DE DIRETÓRIOS ABSOLUTOS DINÂMICOS
    DIRETORIO_SCRIPT = Path(__file__).resolve().parent
    RAIZ_PROJETO = DIRETORIO_SCRIPT.parent

    # Localiza o arquivo YAML procurando na raiz do projeto
    yaml_config = RAIZ_PROJETO / f"data_{args.dataset}.yaml"
    if not yaml_config.exists():
        # Segunda tentativa: procura na pasta atual do terminal
        yaml_config = Path(f"data_{args.dataset}.yaml").resolve()
        if not yaml_config.exists():
            print(f"❌ Erro: O arquivo de configuração {yaml_config} não foi localizado.")
            return

    # Executa a função de detecção que estava faltando
    dispositivo = detectar_dispositivo_treino()

    print(f"\n==============================================================")
    print(f"🎯 Dataset Ativo: {args.dataset.upper()}")
    print(f"📄 Arquivo YAML: {yaml_config}")
    print(f"==============================================================\n")

    # Garante o modelo base apontando para a pasta models na raiz
    modelo_base = RAIZ_PROJETO / "models" / "yolov8s.pt"
    if not modelo_base.exists():
        modelo_base = "yolov8s.pt" # Baixa da nuvem caso não exista localmente

    model = YOLO(str(modelo_base))

    # Definindo pastas de destino absolutas
    pasta_projeto = RAIZ_PROJETO / "models" / "treinamentos"
    nome_treino = f"yolo_{args.dataset}"

    # Dispara o treino com tratamento de workers, caminhos limpos e paciência
    model.train(
        data=str(yaml_config),
        epochs=50,
        imgsz=640,
        batch=16,
        device=dispositivo,
        project=str(pasta_projeto),
        name=nome_treino,
        patience=10,  # Parada antecipada após 10 épocas sem melhora
        workers=0 if os.name == 'nt' else 2  # Evita gargalo de multiprocessamento no Windows
    )

    print(f"\n🔄 Exportando o modelo treinado para o formato ONNX universal...")
    
    # 2. RESOLUÇÃO COMPLETA DO ERRO DE CAMINHO
    caminho_melhor_peso = pasta_projeto / nome_treino / "weights" / "best.pt"
    
    if not caminho_melhor_peso.exists():
        print(f"❌ Erro Crítico: O arquivo de pesos não foi encontrado em: {caminho_melhor_peso}")
        return

    model_treinado = YOLO(str(caminho_melhor_peso))
    
    # Exporta para ONNX (O framework gera o arquivo best.onnx na mesma pasta)
    model_treinado.export(format="onnx", imgsz=640)

    print(f"\n==============================================================")
    print(f"✅ CONCLUÍDO COM SUCESSO!")
    print(f"💾 Modelo PyTorch: {caminho_melhor_peso}")
    print(f"💾 Modelo ONNX:    {caminho_melhor_peso.with_suffix('.onnx')}")
    print(f"==============================================================\n")

# Proteção obrigatória para execução CLI estável no Windows
if __name__ == "__main__":
    treinar_modelo()
