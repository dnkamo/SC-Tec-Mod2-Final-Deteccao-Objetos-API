# VisionGuard 

> Projeto Final Avaliativo — Módulo 2 | Curso SC TEC

### Plataforma Avançada de Visão Computacional Multicenários (YOLOv8s + FastAPI + Streamlit)

O **VisionGuard AI** é um ecossistema completo de Visão Computacional projetado sob os conceitos mais modernos de Engenharia de Software e MLOps. A aplicação opera como um microserviço backend-only de alta performance construído com **FastAPI**, integrado a um Dashboard de Inteligência analítico desenvolvido em **Streamlit**. O sistema conta com um motor híbrido agnóstico a hardware capaz de alternar dinamicamente e em tempo real entre diferentes modelos de inteligência artificial da família **YOLOv8s**.

## Links Globais de Acesso e Planejamento

* **Planejamento Ágil (Quadro Kanban):** `[LINK DO TRELLO DISPONÍVEL APÓS CONCLUSÃO DO BACKLOG]`
* **Demonstração Prática (Vídeo de 7 minutos):** `[LINK DO VÍDEO DO YOUTUBE/DRIVE DISPONÍVEL APÓS GRAVAÇÃO]`
* **Código-Fonte Original:** [Repositório Oficial no GitHub](https://github.com)

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

> **Nota Importante sobre a Infraestrutura de Datasets (.yaml):**
> Para contornar limitações de armazenamento e manter o repositório Git leve, os arquivos de dados brutos não foram indexados na raiz. Os arquivos `data_placas.yaml` e `data_ingredientes.yaml` foram customizados intencionalmente para apontar para caminhos absolutos fora do escopo do Git (armazenados em um drive externo local `E:/projeto dataset/`). 
> Se você clonar este repositório, certifique-se de ajustar a propriedade `path:` dentro dos arquivos `.yaml` para refletir o diretório físico onde as pastas `images/train` e `images/valid` do Roboflow foram extraídas em sua máquina local.

---

## 2. Estrutura de Diretórios do Ecossistema

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

## 3. Tecnologias e Arquitetura

O ecossistema do projeto foi construído utilizando tecnologias estado-da-arte para o ciclo de vida de aplicações de Inteligência Artificial (MLOps) e desenvolvimento de microsserviços. Abaixo estão listadas as ferramentas divididas por suas respectivas camadas de responsabilidade:

### Core de Inteligência Artificial e Processamento Matricial
* **Ultralytics YOLOv8s:** Framework principal utilizado para orquestrar a arquitetura de rede neural convolucional, gerenciamento de pesos, execução de rotinas de Fine-Tuning e exportação multiplataforma.
* **ONNX Runtime (Open Neural Network Exchange):** Mecanismo de inferência de alta performance e agnóstico a hardware. Utilizado como o motor principal no fluxo AMD para extrair predições diretamente de tensores sem dependências de compilação gráfica nativa.
* **OpenCV (Python-NDArray):** Biblioteca de computação científica encarregada da manipulação matricial das imagens (redimensionamento para $640 \times 640$, decodificação de payloads binários e transformações de canais BGR/RGB), desenho geométrico das *Bounding Boxes* e execução manual do algoritmo de Supressão Não-Máxima (`cv2.dnn.NMSBoxes`).
* **PyTorch (Torch-DirectML / CUDA):** Backend matemático responsável pela aceleração de hardware adaptativa em tensores, mapeando dinamicamente instruções para núcleos NVIDIA CUDA ou para a API DirectML do Windows (GPUs AMD/Intel).

### Infraestrutura Backend e Microsserviço
* **FastAPI:** Framework web moderno, assíncrono e de alta performance utilizado para erguer o microsserviço RESTful. Responsável por expor os contratos da API e realizar validações automáticas de tipos e esquemas de dados.
* **Uvicorn (ASGI Server):** Servidor web assíncrono de produção encarregado de servir a aplicação FastAPI sob o protocolo HTTP em ambiente local (`127.0.0.1:8000`).
* **Pydantic / Python-Multipart:** Mecanismo encarregado de interceptar e decodificar fluxos de dados binários complexos (`multipart/form-data`) enviados por clientes remotos, garantindo integridade de tipos no upload.

### Camada de Apresentação e Dashboard Analítico
* **Streamlit:** Framework utilizado para construir a interface visual web de forma declarativa e reativa. Ele gerencia o estado da aplicação em memória (`st.session_state`) e consome a API do FastAPI de forma assíncrona.
* **Requests (HTTP Client):** Biblioteca encarregada de envelopar e despachar os buffers de imagem em memória para os endpoints do backend, gerenciando timeouts e barreira de proxies de rede.

### Auditoria Científica e Engenharia de Dados
* **Pandas:** Utilizado dentro dos ambientes de laboratório para capturar matrizes brutas de validação e convertê-las em tabelas estruturadas de métricas de classificação (*Classification Report*).
* **Seaborn & Matplotlib:** Ferramentas de plotagem gráfica utilizadas nos Jupyter Notebooks para renderizar curvas de aprendizado, decodificar saídas visuais e construir Matrizes de Confusão Reduzidas com filtros dinâmicos de relevância de classes.

---

## 4. Engenharia de IA e Arquitetura Híbrida de Hardware

Para garantir portabilidade e compatibilidade universal nos ambientes locais de desenvolvimento, o motor de inferência foi blindado para operar de forma otimizada em diferentes arquiteturas físicas de computadores:

* **Estratégia de Validação Aberta:** O sistema utiliza o formato **ONNX (Intermediate Representation)** convertido a partir do YOLOv8s. Caso novas rodadas de Fine-Tuning sejam disparadas localmente, os scripts de automação aplicam um downgrade automático de metadados de IR 14 para **IR 13** para garantir compatibilidade com o runtime.
* **Fluxo Exclusivo AMD (Windows DirectML):** Caso o sistema detecte uma GPU AMD/Intel, a aplicação isola o pipeline de execução rodando uma sessão pura de `onnxruntime` em CPUExecutionProvider (contornando travamentos de `inference_mode`), extraindo as tags diretamente dos metadados embutidos e aplicando o algoritmo **NMS (Non-Maximum Suppression)** manualmente via OpenCV (`cv2.dnn.NMSBoxes`) para eliminar duplicidades de caixas.
* **Fluxo Padrão (NVIDIA CUDA / CPU):** Caso uma placa NVIDIA esteja disponível, a inferência roda de forma nativa acelerada por CUDA via SDK da Ultralytics, aplicando o filtro NMS integrado com limiar estrito de interseção (`iou=0.45`).
* **Cache de Modelos em Memória:** A API foi construída com um gerenciador de cache dinâmico. Quando um cenário é solicitado, o arquivo de pesos correspondente é carregado uma única vez e mantido instanciado na memória, garantindo respostas sub-segundo em requisições consecutivas.

---

## 5. Pré-requisitos e Instruções de Instalação (Primeira Execução)

### A. Pré-requisitos do Sistema
Antes de iniciar, certifique-se de ter instalado em sua máquina local:
1. **Git Client:** Para clonar e gerenciar as ramificações do repositório ([Download Oficial](https://git-scm.com)).
2. **Python 3.12.10:** Versão oficial utilizada e homologada para o desenvolvimento deste ecossistema (Garante a compatibilidade estrita das matrizes e dependências de runtime).

---

### B. Passo 1: Clonagem e Preparação do Ambiente Virtual
Abra o seu terminal (Prompt de Comando ou Bash) e execute a sequência abaixo para baixar o projeto e isolar as dependências:

```bash
# 1. Clone o repositório oficial do projeto
git clone https://github.com

# 2. Acesse a pasta raiz do projeto clonado
cd nome-do-seu-repositorio

# 3. Criação do ambiente virtual isolado (.venv)
python -m venv .venv

# 4. Ativação do ambiente virtual
# No Windows (Prompt de Comando):
.venv\Scripts\activate
# No Windows (PowerShell):
.venv\Scripts\Activate.ps1
# No Linux / macOS:
source .venv/bin/activate

# 5. Atualização do gerenciador de pacotes e instalação unificada
pip install --upgrade pip
pip install -r requirements.txt
```

---

### C. Passo 2: Como Posicionar os Pesos Treinados (Preparação de IA)
Como os pesos customizados possuem um tamanho considerável para o histórico padrão do Git, o sistema foi desenhado para ler os arquivos estruturados localmente. Antes de ligar a API, certifique-se de organizar os arquivos `.onnx` exatamente nestes caminhos a partir da raiz do projeto:

* **Modelo Base (Segurança):** `models/yolov8s.onnx`
* **Pesos de Placas de Trânsito:** `models/treinamentos/yolo_placas/weights/best.onnx`
* **Pesos de Ingredientes/Varejo:** `models/treinamentos/yolo_ingredientes/weights/best.onnx`

> **Nota de Fallback:** Caso você inicialize o sistema sem os pesos customizados nas pastas de treinamentos, o arquivo `config.py` disparará um gatilho de segurança que baixa automaticamente o modelo base oficial dos servidores da Ultralytics para a pasta `models/`, permitindo o funcionamento do ambiente sem quebras.

---

### D. Passo 3: Inicialização Sincronizada dos Microsserviços
Para rodar a plataforma completa e habilitar a comunicação assíncrona local, abra **dois terminais distintos** (ambos com o ambiente `.venv` ativado e posicionados na pasta raiz do projeto):

* **Terminal 1 - Inicializar o Backend (FastAPI):**
  ```bash
  uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
  ```
  *A interface interativa e documentada do Swagger ficará disponível em:* **`http://127.0.0.1:8000/detect`**

* **Terminal 2 - Inicializar o Dashboard (Streamlit):**
  ```bash
  streamlit run frontend/app.py
  ```
  *A interface visual amigável abrirá automaticamente no seu navegador em:* **`http://localhost:8501`**

---

### E. Como Executar a Primeira Detecção com Sucesso?
Para homologar o sistema e validar se o pipeline está operando sem barreiras, siga este roteiro de primeiro disparo:

1. **Acesse o Dashboard:** Entre no endereço `http://localhost:8501` no seu navegador.
2. **Selecione a Inteligência:** No menu lateral esquerdo (*Painel de Controle*), utilize a caixa de seleção para escolher o modelo desejado (ex: `🚦 Sinais de Trânsito (Custom)`).
3. **Submeta uma Imagem:** Clique na área central e faça o upload de um arquivo de teste contido na pasta `tests/placas/` (como os arquivos originais do Roboflow com hashes complexos).
4. **Dispare a Inferência:** Clique no botão azul **"Executar Detecção de Objetos"**.
5. **O que esperar do comportamento do sistema?**
   * **Na primeira vez que você seleciona um cenário**, o sistema levará de 2 a 4 segundos para responder, pois a API está carregando os pesos do disco e instanciando o modelo no cache em memória (`self.modelos_carregados`).
   * O terminal do Uvicorn exibirá os logs de carregamento de hardware (identificando sua GPU NVIDIA via CUDA ou AMD via DirectML).
   * A coluna da direita renderizará a imagem final com os **retângulos coloridos e nomes de classes livres de duplicidades** (Graças ao algoritmo NMS), os cards superiores contabilizarão os objetos encontrados e, no rodapé, o **payload JSON estruturado** será exibido.
   * Os disparos seguintes para este mesmo cenário serão executados de forma instantânea (sub-segundo).


---

### F. Como Executar os Notebooks de Auditoria Científica
Caso queira revisar as métricas brutas, gerar o *Classification Report* tabular ou renderizar as Matrizes de Confusão filtradas pelo Seaborn fora da API:

1. Com o ambiente virtual `.venv` ativado, instale o kernel do Jupyter e inicialize o servidor:
   ```bash
   jupyter notebook
   ```
2. No painel que abrir no seu navegador, navegue até a pasta `notebooks/`.
3. Escolha o arquivo correspondente ao seu cenário de teste:
   * **`testes_e_metricas_base.ipynb`**: Para auditar os pesos oficiais de fábrica.
   * **`testes_e_metricas_treinado.ipynb`**: Para auditar os pesos obtidos via Fine-Tuning.
4. Execute as células sequencialmente (`Shift + Enter`). A célula 1 detectará automaticamente o seu hardware (DirectML/CUDA) e aplicará os patches necessários do NumPy para evitar quebras.

---

### G. Como Executar o Script Utilitário de Terminal (`test_yolo_local.py`)
Para realizar uma varredura cega em lote nas imagens extraídas do Roboflow sem precisar ligar o servidor Web:

1. No seu terminal, navegue até o diretório de automações:
   ```bash
   cd scripts
   ```
2. Abra o arquivo `test_yolo_local.py` e configure a variável central `DATASET_ATIVO = "placas"` ou `"ingredientes"` no topo do script.
3. Execute o disparador:
   ```bash
   python test_yolo_local.py
   ```
4. **O que esperar:** O terminal listará a quantidade exata de arquivos complexos com hashes detectados via `os.listdir()`, processará a inferência na GPU e salvará as fotos com os retângulos desenhados de forma isolada dentro da pasta raiz `resultados_teste/[cenario]/`.

---

## 6. Homologação e Validação Isolada da API Backend

Para comprovar a arquitetura limpa de microsserviços (onde o backend funciona de forma 100% independente do Streamlit), o contrato da API FastAPI pode ser testado de forma isolada. Abaixo estão listadas as restrições técnicas, os contratos de rede e as formas de validação do endpoint.

### A. Restrições Técnicas do Endpoint `POST /detect`
* **Formatos de Imagem Permitidos:** O microsserviço aplica uma barreira de segurança estrita por extensão de arquivo. São aceitos estritamente os formatos binários: `.jpg`, `.jpeg`, `.png`, `.bmp` e `.webp`. Enviar qualquer outro formato disparará uma exceção tratada imediatamente.
* **SLAs e Tempo de Execução Esperado:**
  * **Primeira Requisição (Frio/Cold Start):** De `2.0` a `4.0` segundos. Esse tempo é computado apenas no primeiro disparo de cada cenário, pois a API precisa ler o arquivo `.onnx` do disco, validar os metadados IR e alocar a sessão do modelo no cache de memória.
  * **Requisições Consecutivas (Quente/Warm Start):** De `0.15` a `0.55` segundos (sub-segundo). Graças ao gerenciamento de cache da classe `YoloService`, o modelo responde de forma quase instantânea a partir da memória RAM/VRAM.

### B. Matriz de Códigos de Retorno HTTP Tratados
O microsserviço utiliza os padrões HTTP semânticos para responder sobre a integridade da transação:
* **`200 OK`:** A imagem foi recebida, a inferência foi executada e o JSON estruturado foi devolvido com sucesso.
* **`400 Bad Request`:** Extensão de arquivo inválida. Ocorre quando o cliente tenta enviar arquivos não suportados (ex: `.pdf`, `.gif`, `.txt`).
* **`422 Unprocessable Entity`:** Imagem corrompida ou inválida. Ocorre se o arquivo enviado estiver quebrado e a matriz OpenCV (`cv2.imdecode`) não conseguir convertê-lo em tensores numéricos.
* **`500 Internal Server Error`:** Falha interna inesperada no pipeline de inferência da IA ou colisão de hardware de baixo nível.

---

### C. Método 1: Validação Visual via Swagger UI (Nativo)
1. Certifique-se de que o Uvicorn está rodando e acesse: **`http://127.0.0.1:8000/detect`**
2. Expanda o bloco do endpoint **`POST /detect`**.
3. Clique no botão **"Try it out"** no canto direito.
4. No parâmetro `cenario`, escolha entre `placas`, `ingredientes` ou `padrao` no menu dropdown interativo.
5. No campo `file`, clique em *Escolher arquivo* e selecione uma imagem de teste válida (`.jpg` ou `.png`).
6. Clique no botão azul **"Execute"**. O painel exibirá o código de retorno `200` e o JSON estruturado exatamente igual ao exigido na página 3 do edital.

### D. Método 2: Validação via Terminal (Comando cURL)
Você pode simular uma integração direta de sistemas disparando um pacote `multipart/form-data` direto via terminal usando utilitários como o `cURL` (passando o parâmetro de query correspondente ao cenário). Substitua `[cenario]` por `placas` ou `ingredientes` e aponte para um arquivo real local:

```bash
# Teste de envio de imagem local para o cenário de placas
curl -X 'POST' \
  'http://127.0.0.1:8000/detect?cenario=[cenario]' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@tests/placas/seu_arquivo_teste.jpg;type=image/jpeg'
```
Assim, o microsserviço devolve dados puros e organizados de geolocalização dos objetos, classes e taxas de confiança.

## 7. Exemplo de Resposta JSON
Abaixo estão os formatos de payloads reais retornados pelo endpoint de acordo com o modelo selecionado pelo usuário:

#### Cenário A: Sinais de Trânsito (`cenario=placas`)
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

#### Cenário B: Varejo/Vegetais (`cenario=ingredientes`)
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

### C. Descrição Técnica dos Campos Retornados
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

## 8. Análise de Métricas, Benchmarks e Resultados

Esta seção documenta a avaliação estatística rigorosa do comportamento dos modelos em três cenários distintos, estabelecendo um comparativo claro entre a inteligência de fábrica (Baseline COCO) e os pesos customizados pós-treinamento (*Fine-Tuning*).

---

### A. O Cenário de Fallback: Modelo Baseline Padrão (`yolov8s`)
* **Contexto de Teste:** O modelo original de fábrica (treinado no dataset COCO com 80 objetos genéricos) foi submetido a uma varredura cruzada utilizando as imagens específicas de placas e ingredientes.
* **Resultado Global:** 
  * **Precisão Média:** `0.0066` (Menos de 1%)
  * **Revocação Média:** `0.1168` (11.6%)
  * **mAP50 Geral:** `0.0046`

#### Tabela de Amostragem do Relatório de Validação (Baseline COCO)

| Classe Alvo | Precisão (Precision) | Revocação (Recall) | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: |
| `person` | 0.0009 | 0.0333 | 0.0006 | 0.0001 |
| `car` | 0.0103 | 0.2647 | 0.0066 | 0.0049 |
| `motorcycle` | 0.0123 | 0.0645 | 0.0074 | 0.0047 |
| `traffic light` | 0.0050 | 0.0833 | 0.0030 | 0.0019 |
| **MÉDIA GLOBAL (all)** | **0.0066** | **0.1168** | **0.0046** | **0.0025** |

> **Justificativa de IA do Projeto:** Como evidenciado pelos números e pela Matriz de Confusão do Baseline, os objetos do modelo genérico entram em colapso completo contra os dados reais do Roboflow (as detecções de trânsito e vegetais caem no ruído de fundo ou são rotuladas erroneamente como `person` ou `car`). Isso comprova cientificamente que **apenas rodar o modelo padrão de fábrica é insuficiente**, justificando a necessidade imediata das rotinas de Fine-Tuning do projeto.

---

### B. Cenário Customizado 1: Sinais de Trânsito (`data_placas.yaml`)
* **Hardware Utilizado:** Processamento em lote de 488 imagens / 529 instâncias rotuladas.
* **Resultado Global pós Fine-Tuning:**
  * **Precisão Média (Precision):** `0.9475` (94.7% de assertividade nos alarmes dados)
  * **Revocação Média (Recall):** `0.9496` (94.9% de capacidade de localizar as placas na cena)
  * **mAP50 Geral:** `0.9650` (96.5% de precisão geral na geometria das caixas)
  * **mAP50-95 Global:** `0.8230`

#### Classification Report - Top Classes (Placas de Trânsito)

| Classe | Precisão (Precision) | Revocação (Recall) | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: |
| `bus_stop` | 0.9579 | 1.0000 | 0.9950 | 0.9321 |
| `do_not_enter` | 0.8982 | 0.9333 | 0.9641 | 0.9069 |
| `do_not_stop` | 0.9710 | 0.9865 | 0.9936 | 0.9040 |
| `green_light` | 0.9320 | 1.0000 | 0.9950 | 0.9734 |
| `stop` | 0.9895 | 1.0000 | 0.9950 | 0.8587 |
| **MÉDIA GLOBAL (all)** | **0.9475** | **0.9496** | **0.9650** | **0.8230** |

#### Análise Gráfica (Matriz de Confusão - Placas)
A análise da Matriz de Confusão Reduzida do Seaborn comprova uma convergência diagonal quase perfeita. As classes críticas de segurança viária como `do_not_enter` (30 acertos puros) e `stop` (30 acertos puros) obtiveram erro zero contra o background. O modelo apresentou leves confusões aceitáveis de falso-negativo estritamente concentradas entre variações sutis de semáforos (`red_light` e `yellow_light` confundidos com o background devido à escala de luminosidade das fotos).

---

### Cenário Customizado 2: Varejo e Alimentos (`data_ingredientes.yaml`)
* **Hardware Utilizado:** Processamento de 582 imagens contendo uma alta densidade de alvos (1794 instâncias).
* **Resultado Global pós Fine-Tuning:**
  * **Precisão Média (Precision):** `0.6734`
  * **Revocação Média (Recall):** `0.5462`
  * **mAP50 Geral:** `0.5908`
  * **mAP50-95 Global:** `0.3763`

#### Classification Report - Top Classes (Ingredientes/Mercado)

| Classe | Precisão (Precision) | Revocação (Recall) | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: |
| `onion` | 0.8766 | 0.7206 | 0.8152 | 0.5726 |
| `cayliflower` | 0.8743 | 0.7249 | 0.8158 | 0.5196 |
| `avocado` | 0.8655 | 0.7089 | 0.7899 | 0.4891 |
| `corn` | 0.8442 | 0.7407 | 0.8305 | 0.5041 |
| `tomato` | 0.5889 | 0.6957 | 0.6147 | 0.4153 |
| **MÉDIA GLOBAL (all)** | **0.6734** | **0.5462** | **0.5908** | **0.3763** |

#### Análise Gráfica (Matriz de Confusão - Ingredientes)
Diferente do cenário de tráfego, o dataset de ingredientes lida com **alta oclusão e deformação biológica** (vegetais amontoados em gôndolas). A Matriz de Confusão Reduzida do Seaborn revela alta performance em itens com texturas e formas bem definidas como `tomato` (79 acertos), `bell pepper` (70 acertos) e `garlic` (66 acertos). Os gargalos de métricas concentraram-se na classe `beans` (apenas 10 acertos), que sofreu confusão direta com o `background` (38 instâncias perdidas) devido ao tamanho reduzido do grão e à camuflagem cromática em relação aos recipientes nas fotos.

---

## 9. Ferramentas Utilitárias e de Laboratório

O repositório conta com subdiretórios específicos para controle de qualidade, auditorias de inteligência artificial e treinamentos em lote por linha de comando:

* **`notebooks/testes_e_metricas_base.ipynb`:** Jupyter Notebook automatizado responsável por carregar o modelo baseline de fábrica (`yolov8s.onnx`). Ele utiliza a biblioteca `glob` para varrer dinamicamente as pastas de amostras e o dicionário centralizado `CONFIG_DATASETS` para carregar de forma inteligente o cenário ativo (`placas` ou `ingredientes`). Gera o Classification Report tabular e as Matrizes de Confusão (Completa e Reduzida Filtrada via Seaborn) forçando execução em CPU se o hardware ativo for AMD para evitar colisões de tensores.
* **`notebooks/testes_e_metricas_treinado.ipynb`:** Jupyter Notebook com arquitetura idêntica ao de base, porém parametrizado para apontar obrigatoriamente para os arquivos de pesos customizados (`best.onnx`) gerados na pasta `models/treinamentos/` após a conclusão das rotinas de Fine-Tuning. Ideal para demonstrar o ganho estatístico de acertos em células interativas no vídeo.
* **`scripts/test_yolo_local.py`:** Script utilitário automatizado de terminal. Ele utiliza a função `os.listdir()` para varrer de forma cega as pastas locais do Roboflow (contornando automaticamente as strings complexas e hashes de nomes de arquivos), desenhando as bounding boxes via OpenCV e organizando os resultados físicos nas subpastas correspondentes em `resultados_teste/`.
* **`scripts/train_yolo.py`:** Script dinâmico que aceita argumentos de linha de comando (`--dataset placas` ou `--dataset ingredientes`), configura automaticamente os workers para sistemas Windows e dispara o treinamento injetando o hardware adaptativo.


---

Desenvolvido como atividade prática avaliativa do Módulo 2 (Visão Computacional e Aprendizado Profundo) do curso SC TEC.