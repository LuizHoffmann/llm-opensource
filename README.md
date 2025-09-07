# Assistente Virtual Acadêmico - PESC/Coppe 📜

## 📖 Descrição
Este projeto é um agente de IA (ChatBot) desenvolvido para auxiliar alunos do Programa de Engenharia de Sistemas e Computação (PESC) da Coppe/UFRJ a sanar dúvidas sobre regulamentos acadêmicos.

Utilizando uma arquitetura de **Geração Aumentada por Recuperação (RAG)**, o agente é capaz de entender a pergunta do usuário em linguagem natural, identificar autonomamente os documentos relevantes, e formular uma resposta precisa baseada estritamente nas fontes de conhecimento fornecidas.

---

## ✨ Funcionalidades
- **Roteamento Inteligente**: O agente analisa a pergunta e decide qual base de conhecimento (bolsas, regras ou defesas) é a mais adequada para a busca.
- **Geração de Respostas Baseada em Contexto**: As respostas são geradas pelo modelo **GPT-4o da Azure OpenAI** com base nos trechos extraídos dos documentos oficiais.
- **Citação de Fontes**: Para cada resposta, o assistente fornece os trechos exatos dos documentos que utilizou, garantindo transparência e confiabilidade.
- **Interface Interativa**: Uma interface web simples e intuitiva construída com **Streamlit**.

---

## 🏗️ Arquitetura
O projeto é modular e utiliza tecnologias de ponta para orquestração de IA e processamento de linguagem.

### Frontend
- **Streamlit**: Biblioteca Python utilizada para criar a interface web do chat de forma rápida e interativa.

### Backend
- **Orquestrador de Agente (LangGraph)**: Define o fluxo de trabalho lógico do agente. O grafo direciona a pergunta do usuário através de nós especializados: roteamento, busca de documentos e geração de resposta.
- **Modelos de IA (Azure OpenAI)**:
  - `gpt-4o`: Utilizado tanto para rotear a pergunta do usuário quanto para gerar a resposta final.
  - `text-embedding-ada-002`: Responsável por converter os documentos de texto em vetores numéricos (embeddings) para a busca semântica.
- **Banco de Dados Vetorial (Qdrant Cloud)**: Armazena os vetores dos documentos e permite buscas de similaridade em alta velocidade para encontrar os trechos mais relevantes.

---

## 🚀 Configuração e Execução

### Pré-requisitos
- Python **3.10** (recomendado para compatibilidade)
- Git

### 1. Clone o Repositório
```bash
git clone git@github.com:LuizHoffmann/llm-opensource.git
cd llm-opensource
```

### 2. Crie e Ative um Ambiente Virtual
#### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

> **Nota**: Se você tiver múltiplas versões de Python instaladas, pode ser necessário especificar o caminho completo para o executável do Python 3.10.  
> Exemplo no Windows:  
> `C:\Users\SEU_USUARIO\AppData\Local\Programs\Python\Python310\python.exe -m venv .venv`

### 3. Instale as Dependências
```bash
pip install -r requirements.txt
```

### 4. Configure as Variáveis de Ambiente
Crie um arquivo chamado `.env` na raiz do projeto com o seguinte conteúdo:

```bash
# Credenciais da Azure OpenAI
AZURE_OPENAI_API_KEY="SUA_CHAVE_API_AZURE"
AZURE_OPENAI_ENDPOINT="SEU_ENDPOINT_AZURE"
AZURE_OPENAI_API_TYPE="azure"

# Modelos de IA (Deployments)
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="gpt-4o"
EMBEDDING_MODEL_DEPLOYMENT_NAME="text-embedding-ada-002"

# Versões da API
AZURE_OPENAI_API_VERSION_LLM="2024-02-01"
AZURE_OPENAI_API_VERSION_EMBEDDING="2023-05-15"

# Credenciais do Qdrant Cloud
QDRANT_URL="SUA_URL_DO_CLUSTER_QDRANT"
QDRANT_API_KEY="SUA_CHAVE_API_QDRANT"
```

### 5. Indexe os Documentos (Ingestão de Dados)
Antes de executar a aplicação pela primeira vez, você precisa processar seus documentos e enviá-los para o Qdrant:

```bash
python -m src.llm.rag
```

> Esse passo só precisa ser executado uma vez ou sempre que os documentos na pasta `/docs` forem alterados.

### 6. Execute a Aplicação
```bash
streamlit run main.py
```
O aplicativo será aberto automaticamente no navegador.

---

## 📁 Estrutura do Projeto
```
├── .venv/
├── docs/
├── src/
│   ├── llm/
│   │   ├── agent.py          # Lógica do agente com LangGraph (cérebro da aplicação)
│   ├── rag/
│   │   └── rag.py            # Script para ingestão e indexação de dados
│   └── utils/
│       └── logger_config.py  # Configuração do sistema de logs
├── .env
├── .gitignore
├── app.log
├── main.py                   # Ponto de entrada e interface com Streamlit
├── README.md
└── requirements.txt
```
