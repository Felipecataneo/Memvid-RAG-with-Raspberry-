# RAG Local com Memvid e Ollama no Raspberry Pi

Este projeto implementa um sistema de **Retrieval-Augmented Generation (RAG)** completo, projetado para rodar de forma 100% local e eficiente em um Raspberry Pi. Ele utiliza a biblioteca **Memvid** para criar uma base de conhecimento compacta a partir de documentos PDF, e o **Ollama** com o modelo **gemma3** para gerar respostas.

O objetivo é ter um assistente pessoal, privado e inteligente, capaz de responder perguntas com base nos seus próprios documentos.

## Como Funciona?

A inovação central deste projeto é o uso do **memvid** para gerenciar a base de conhecimento. Em vez de um banco de dados vetorial tradicional que consome muita RAM, este sistema:

- Extrai o texto de todos os arquivos PDF que você fornecer
- Codifica o texto e seus embeddings vetoriais diretamente nos pixels de um arquivo de vídeo (usando o codec mp4v, que é leve e nativo)
- Gera arquivos de índice (`.json` e `.faiss`) para permitir buscas rápidas

Isso resulta em uma base de conhecimento extremamente portátil e com baixo consumo de memória, ideal para dispositivos como o Raspberry Pi. A API, construída com **FastAPI**, gerencia as buscas e a interação com o modelo de linguagem local.

## Arquitetura do Projeto

O fluxo de trabalho é dividido em duas partes:

### 1. Construção da Memória (`build_memory_from_pdfs.py`)
- Lê os arquivos `.pdf` da pasta `papers/`
- Cria chunks de texto e gera embeddings
- Salva tudo em `output/` como `memory.mp4v` e seus índices

### 2. Servidor de Chat (`main.py`)
- Inicia uma API FastAPI otimizada
- Carrega a memória memvid
- Recebe perguntas via HTTP
- Busca os chunks mais relevantes na memória
- Monta um prompt com o contexto e envia para o modelo gemma3 no Ollama
- Retorna a resposta gerada pelo modelo

## Recursos

- **100% Local e Privado**: Seus documentos e suas perguntas nunca saem da sua máquina
- **Otimizado para Raspberry Pi**: Código ajustado para baixo consumo de CPU e RAM, com cache inteligente, timeouts e configurações de concorrência adequadas
- **Modelo Leve**: Pré-configurado para `gemma3:1b`, um modelo poderoso e eficiente
- **Fácil de Usar**: Apenas dois scripts para construir a memória e iniciar o serviço
- **Gerenciamento de Contexto**: Otimiza o número de chunks enviados ao LLM para evitar exceder a janela de contexto e maximizar a relevância
- **Portátil**: Faça backup ou mova sua base de conhecimento simplesmente copiando a pasta `output/`

## Pré-requisitos

Antes de começar, garanta que você tenha:

- **Hardware**: Um Raspberry Pi 4 ou 5 (8GB de RAM é recomendado)
- **Sistema**: Python 3.11 ou superior instalado
- **Ollama**: O serviço do Ollama deve estar instalado e em execução
- **Modelo LLM**: O modelo `gemma3:1b` precisa estar baixado. Se não tiver, execute:

```bash
ollama pull gemma3:1b
```

## Instalação

1. **Clone este repositório:**
```bash
git clone https://github.com/Felipecataneo/Memvid-RAG-with-Raspberry.git
cd Memvid-RAG-with-Raspberry
```

2. **Crie e ative um ambiente virtual (altamente recomendado):**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Instale todas as dependências:**
```bash
pip install -r requirements.txt
```

## Como Usar

O processo tem duas etapas simples:

### Passo 1: Construir a Memória com Seus Documentos

1. Se a pasta `papers/` não existir, crie-a na raiz do projeto
2. Copie todos os seus arquivos `.pdf` para dentro da pasta `papers/`
3. Execute o script de construção da memória:

```bash
python build_memory_from_pdfs.py
```

O script irá processar cada PDF e, ao final, criará os seguintes arquivos na pasta `output/`:
- `memory.mp4v`
- `memory_index.json`
- `memory_index.faiss`

### Passo 2: Iniciar a API e Conversar

1. Com a memória construída, inicie o servidor FastAPI:
```bash
python main.py
```

2. A API estará rodando em `http://0.0.0.0:8000`

3. Agora você pode enviar perguntas para seus documentos. Use uma ferramenta como `curl` ou qualquer cliente de API.

**Exemplo de consulta com curl:**
```bash
curl -X POST http://localhost:8000/chat \
-H "Content-Type: application/json" \
-d '{
  "query": "Qual a principal metodologia descrita nos documentos?",
  "use_cache": true,
  "detail_level": "medium"
}'
```

### Parâmetros da consulta:

- **`query`**: A pergunta que você quer fazer
- **`use_cache`**: `true` para usar o cache de respostas (mais rápido para perguntas repetidas)
- **`detail_level`**: Controle o nível de detalhe da resposta. Pode ser:
  - `"brief"`: Resposta curta e direta
  - `"medium"`: Resposta balanceada
  - `"detailed"`: Resposta aprofundada

## Endpoints da API

- **`POST /chat`**: Endpoint principal para fazer perguntas
- **`GET /health`**: Verifica o status da API e se a memória foi carregada
- **`GET /stats`**: Exibe estatísticas sobre a memória e o cache
- **`DELETE /cache`**: Limpa o cache de respostas

## Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests.

## Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
