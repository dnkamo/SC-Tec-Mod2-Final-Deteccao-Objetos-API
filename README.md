# VisionGuard AI

### Plataforma Avançada de Visão Computacional Multicenários (YOLOv8s + FastAPI + Streamlit)

O **VisionGuard AI** é um ecossistema completo de Visão Computacional projetado sob os conceitos mais modernos de Engenharia de Software e MLOps. A aplicação opera como um microserviço backend-only de alta performance construído com **FastAPI**, integrado a um Dashboard de Inteligência analítico desenvolvido em **Streamlit**. O sistema conta com um motor híbrido agnóstico a hardware capaz de alternar dinamicamente e em tempo real entre diferentes modelos de inteligência artificial da família **YOLOv8s**.

---

## 1. O Desafio e os Datasets Escolhidos

O sistema foi arquitetado para resolver problemas complexos de detecção e localização de objetos em múltiplos domínios comerciais e industriais de forma dinâmica. O ecossistema suporta nativamente o mapeamento, carregamento e a alternância instantânea entre dois cenários públicos reais hospedados na plataforma Roboflow Universe:

* **Cenário 1: Tráfego e Sinais de Trânsito (`data_placas.yaml`)**
  * **Dataset Oficial:** [Road Signs - Roboflow 100](https://universe.roboflow.com/roboflow-100/road-signs-6ih4y)
  * **Desafio:** Auxiliar sistemas de monitoramento automotivo, mapeamento de vias e direção autônoma.
  * **Escopo (21 Classes):** `bus_stop`, `do_not_enter`, `do_not_stop`, `do_not_turn_l`, `do_not_turn_r`, `do_not_u_turn`, `enter_left_lane`, `green_light`, `left_right_lane`, `no_parking`, `parking`, `ped_crossing`, `ped_zebra_cross`, `railway_crossing`, `red_light`, `stop`, `t_intersection_l`, `traffic_light`, `u_turn`, `warning`, `yellow_light`.

* **Cenário 2: Varejo e Alimentos/Mercado (`data_ingredientes.yaml`)**
  * **Dataset Oficial:** [Ingredient Detection Dataset](https://universe.roboflow.com/object-detection-yolo-feykl/ingredient-detection-kzuti)
  * **Desafio:** Controle de estoque automatizado em gôndolas e sistemas de self-checkout em supermercados.
  * **Escopo (26 Classes):** `avocado`, `beans`, `beet`, `bell pepper`, `broccoli`, `brus capusta`, `cabbage`, `carrot`, `cayliflower`, `celery`, `corn`, `cucumber`, `eggplant`, `fasol`, `garlic`, `hot pepper`, `onion`, `peas`, `potato`, `pumpkin`, `rediska`, `redka`, `salad`, `squash-patisson`, `tomato`, `vegetable marrow`.

---

## 📂 2. Estrutura de Diretórios do Ecossistema

O repositório foi organizado seguindo as melhores práticas de arquitetura modular de software e isolamento de escopo (Separação de Conceitos). Arquivos massivos de imagens de treino e arquivos pesados binários de pesos locais não são versionados, mantendo o repositório leve e focado em código-fonte:

```text
pasta-raiz/
│
├── .gitignore                 # Filtra venv, caches locais e barra uploads de arquivos pesados (*.onnx/*.pt)
├── README.md                  # Documentação técnica completa, payloads e instruções de execução
├── requirements.txt           # Gerenciador unificado de dependências com versões travadas para produção
│
├── data_placas.yaml           # Configuração de classes e rotas absolutas do dataset de trânsito (Roboflow)
├── data_ingredientes.yaml     # Configuração de classes e rotas absolutas do dataset de vegetais (Roboflow)
│
├── app/                       # MICROSSERVIÇO BACKEND (API FastAPI)
│   ├── __init__.py
│   ├── main.py                # Inicializador oficial do FastAPI, ponto de partida Uvicorn e regras estritas de CORS
│   ├── config.py              # Varredura automatizada de hardware e correção de metadados IR do ONNX
│   │
│   ├── api/                   # Camada de Entrada (Interface de Comunicação HTTP)
│   │   ├── __init__.py
│   │   └── endpoints.py       # Endpoint POST /detect que recebe e valida o UploadFile binário
│   │
│   └── services/              # Camada de Regras de Negócio (Abstração da IA)
│       ├── __init__.py
│       └── yolo_service.py    # Motor híbrido e agnóstico (InferenceSession AMD vs SDK Ultralytics NVIDIA) + Filtro NMS
│
├── frontend/                  # INTERFACE GRÁFICA WEB (Dashboard Streamlit)
│   └── app.py                 # Dashboard analítico com st.session_state estável e exportação de relatórios
│
├── models/                    # ARMAZENAMENTO DE PESOS E CONTINGÊNCIAS (Diretório local)
│   ├── yolov8s.onnx           # Peso baseline de fábrica convertido para o formato universal ONNX
│   └── treinamentos/          # Subpastas geradas de forma automatizada pelas rotas de Fine-Tuning
│       ├── yolo_placas/       # Contém o arquivo best.onnx de alta performance para sinais de trânsito
│       └── yolo_ingredientes/ # Contém o arquivo best.onnx de alta performance para vegetais
│
├── notebooks/                 # AMBIENTE DIDÁTICO E AUDITORIAS DE IA (Para gravação do vídeo)
│   ├── testes_e_metricas_base.ipynb       # Células sequenciais para avaliação do modelo de fábrica padrão
│   └── testes_e_metricas_treinado.ipynb   # Células sequenciais para extração do Classification Report pós-treino
│
├── scripts/                   # SCRIPTS UTILITÁRIOS E AUTOMAÇÕES VIA TERMINAL
│   ├── test_yolo_local.py     # Script com os.listdir() cego para testar imagens Roboflow em lote local
│   └── train_yolo.py          # Disparador adaptativo de treinamento acoplado ao DirectML/CUDA e exportação ONNX
│
└── tests/                     # DIRETÓRIO LOCAL DE AMOSTRAS DE VALIDAÇÃO
    ├── placas/                # Imagens com hashes complexos do Roboflow para teste de trânsito
    └── ingredientes/          # Imagens com hashes complexos do Roboflow para teste de varejo
```

---

## 2. Engenharia de IA e Arquitetura Híbrida de Hardware

Para garantir portabilidade e compatibilidade universal nos ambientes locais de desenvolvimento, o motor de inferência foi blindado para operar de forma otimizada em diferentes arquiteturas físicas de computadores:

* **Estratégia de Validação Aberta:** O sistema utiliza o formato **ONNX (Intermediate Representation)** convertido a partir do YOLOv8s. Caso novas rodadas de Fine-Tuning sejam disparadas localmente, os scripts de automação aplicam um downgrade automático de metadados de IR 14 para **IR 13** para garantir compatibilidade com o runtime.
* **Fluxo Exclusivo AMD (Windows DirectML):** Caso o sistema detecte uma GPU AMD/Intel, a aplicação isola o pipeline de execução rodando uma sessão pura de `onnxruntime` em CPUExecutionProvider (contornando travamentos de `inference_mode`), extraindo as tags diretamente dos metadados embutidos e aplicando o algoritmo **NMS (Non-Maximum Suppression)** manualmente via OpenCV (`cv2.dnn.NMSBoxes`) para eliminar duplicidades de caixas.
* **Fluxo Padrão (NVIDIA CUDA / CPU):** Caso uma placa NVIDIA esteja disponível, a inferência roda de forma nativa acelerada por CUDA via SDK da Ultralytics, aplicando o filtro NMS integrado com limiar estrito de interseção (`iou=0.45`).
* **Cache de Modelos em Memória:** A API foi construída com um gerenciador de cache dinâmico. Quando um cenário é solicitado, o arquivo de pesos correspondente é carregado uma única vez e mantido instanciado na memória, garantindo respostas sub-segundo em requisições consecutivas.

---

## 3. Instruções de Instalação e Execução Local

### Passo 1: Preparação do Ambiente Virtual e Instalação
Abra o seu terminal na pasta raiz do projeto e execute os comandos correspondentes ao seu Sistema Operacional:

```bash
# 1. Criação do ambiente virtual isolado
python -m venv .venv

# 2. Ativação do ambiente
# No Windows (Prompt de Comando):
.venv\Scripts\activate
# No Linux/macOS:
source .venv/bin/activate

# 3. Atualização do gerenciador e instalação das dependências unificadas
pip install --upgrade pip
pip install -r requirements.txt
```

### 🚀 Passo 2: Inicialização Sincronizada dos Microsserviços
Para rodar a plataforma completa e habilitar a comunicação assíncrona local entre os sistemas, você precisará abrir **dois terminais distintos** (ambos com o `.venv` ativado e posicionados na raiz do projeto):

* **Terminal 1 - Inicializar o Backend (FastAPI):**
  ```bash
  uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
  ```
  *A interface interativa e documentada do Swagger ficará disponível imediatamente em:* **`http://127.0.0.1:8000/detect`**

* **Terminal 2 - Inicializar o Dashboard (Streamlit):**
  ```bash
  streamlit run frontend/app.py
  ```
  *A interface visual amigável do cliente abrirá automaticamente no seu navegador em:* **`http://localhost:8501`**

---

## 4. Contratos de Interface (Estrutura do JSON e Exemplos de Uso)

O microsserviço devolve dados puros e organizados de geolocalização dos objetos, classes e taxas de confiança.

### A. Exemplo de Requisição (Teste Manual via Terminal / Integração)
Além de testar pela interface visual do Streamlit ou pelo Swagger, você pode realizar requisições multipart/form-data manuais via terminal usando utilitários como o `cURL` (passando o parâmetro de query correspondente ao cenário):

```bash
# Teste de envio de imagem local para o cenário de placas
curl -X 'POST' \
  'http://127.0.0.1:8000/detect' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@caminho/para/sua/imagem_teste.jpg;type=image/jpeg'
```

### B. Exemplo de Resposta JSON
Abaixo estão os formatos de payloads reais retornados pelo endpoint de acordo com o modelo selecionado pelo usuário:

#### Cenário: Sinais de Trânsito (`cenario=placas`)
```json
{
  "status": "success",
  "cenario_solicitado": "placas",
  "cenario_executado": "placas",
  "detections": [
    {
      "class_name": "stop",
      "confidence": 0.94,
      "box": {
        "x_min": 142,
        "y_min": 210,
        "x_max": 285,
        "y_max": 350
      }
    }
  ]
}
```

#### 🥦 Cenário: Varejo/Vegetais (`cenario=ingredientes`)
```json
{
  "status": "success",
  "cenario_solicitado": "ingredientes",
  "cenario_executado": "ingredientes",
  "detections": [
    {
      "class_name": "tomato",
      "confidence": 0.88,
      "box": {
        "x_min": 45,
        "y_min": 89,
        "x_max": 120,
        "y_max": 195
      }
    }
  ]
}
```

### 📋 C. Descrição Técnica dos Campos Retornados
* **`status`** `(string)`: Indica o estado da transação da inferência (`success` se a imagem foi processada corretamente ou `error` se a matriz de imagem estiver inválida ou corrompida).
* **`cenario_solicitado`** `(string)`: O modelo que o cliente selecionou via parâmetro de busca (`placas`, `ingredientes` ou `padrao`).
* **`cenario_executado`** `(string)`: Exibe o modelo que realmente realizou o processamento em memória. Útil para auditorias, pois evidencia se o sistema ativou o arquivo de pesos customizados ou se precisou aplicar a contingência de *Fallback* automático (`padrao (fallback)`) devido à ausência física do arquivo de pesos local.
* **`detections`** `(array)`: Lista aninhada contendo todos os objetos capturados acima do limiar de corte de 40% (Filtro NMS).
  * **`class_name`** `(string)`: Nome mapeado da classe identificada (nativamente extraído do arquivo `.yaml` correspondente).
  * **`confidence`** `(float)`: Taxa de acerto estatístico da inferência (Varia de `0.00` a `1.00`, arredondado para duas casas decimais).
  * **`box`** `(object)`: Dicionário contendo as coordenadas da caixa delimitadora (*Bounding Box*) mapeadas nos pixels reais da imagem original do cliente:
    * **`x_min`** `(int)`: Coordenada do pixel do canto superior esquerdo (Eixo X horizontal).
    * **`y_min`** `(int)`: Coordenada do pixel do canto superior esquerdo (Eixo Y vertical).
    * **`x_max`** `(int)`: Coordenada do pixel do canto inferior direito (Eixo X horizontal).
    * **`y_max`** `(int)`: Coordenada do pixel do canto inferior direito (Eixo Y vertical).

---

## 5. Ferramentas Utilitárias e de Laboratório

O repositório conta com subdiretórios específicos para controle de qualidade, auditorias de inteligência artificial e treinamentos em lote por linha de comando:

* **`notebooks/testes_e_metricas_base.ipynb`:** Jupyter Notebook responsável por carregar o baseline, executar inferências em células isoladas e gerar o relatório tabular consolidado (*Classification Report* via Pandas) e a Matriz de Confusão Reduzida gerada dinamicamente filtrando as Top 10 classes mais relevantes pelo Seaborn.
* **`notebooks/testes_e_metricas_base.ipynb`:** Jupyter Notebook dedicado a auditar os arquivos de pesos customizados (`best.onnx`) após as execuções de Fine-Tuning.
* **`scripts/test_yolo_local.py`:** Script utilitário automatizado de terminal. Ele utiliza a função `os.listdir()` para varrer de forma cega as pastas locais do Roboflow (contornando automaticamente as strings complexas e hashes de nomes de arquivos), desenhando as bounding boxes via OpenCV e organizando os resultados físicos nas subpastas correspondentes em `resultados_teste/`.
* **`scripts/train_yolo.py`:** Script dinâmico que aceita argumentos de linha de comando (`--dataset placas` ou `--dataset ingredientes`), configura automaticamente os workers para sistemas Windows e dispara o treinamento injetando o hardware adaptativo.
