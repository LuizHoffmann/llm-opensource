# --- Importe as bibliotecas necessárias ---
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient

# --- Configurações Iniciais ---
# Endereço do seu servidor Qdrant
QDRANT_URL = "http://localhost:6333"

# Modelo de embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Lista dos documentos
documentos_para_indexar = [
    {
        "caminho": "./docs/BolsasAcademicas.txt",
        "colecao": "bolsas_academicas",
        "tipo": "txt",
    },
    {
        "caminho": "./docs/RegrasCOPPE2020.pdf",
        "colecao": "regras_coppe_2020",
        "tipo": "pdf",
    },
    {
        "caminho": "./docs/RoteiroDefesaRemota.pdf",
        "colecao": "roteiro_defesa_remota",
        "tipo": "pdf",
    },
]

# --- 1. Carregando, Processando e Indexando os Documentos (com verificação) ---
print("--- Iniciando a verificação e criação dos índices no Qdrant ---")

qdrant_client_admin = QdrantClient(url=QDRANT_URL)

for doc_info in documentos_para_indexar:
    caminho = doc_info["caminho"]
    colecao = doc_info["colecao"]
    tipo = doc_info["tipo"]

    try:
        if qdrant_client_admin.collection_exists(collection_name=colecao):
            print(f"Índice '{colecao}' já existe. Pulando a criação.")
            continue
    except Exception as e:
        print(f"Não foi possível conectar ao Qdrant. Verifique se o servidor está no ar. Erro: {e}")
        break

    print(f"Criando índice para a coleção '{colecao}'...")
    try:
        if not os.path.exists(caminho):
            print(f"AVISO: Arquivo não encontrado em '{caminho}'. Pulando.")
            continue

        if tipo == "pdf":
            loader = PyPDFLoader(caminho)
        elif tipo == "txt":
            loader = TextLoader(caminho)
        else:
            print(f"Tipo de arquivo não suportado: {caminho}")
            continue

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        Qdrant.from_documents(
            docs,
            embeddings,
            url=QDRANT_URL,
            collection_name=colecao,
        )
        print(f"Índice '{colecao}' criado com sucesso!")

    except Exception as e:
        print(f"Erro ao processar o arquivo {caminho}: {e}")

print("\n--- Verificação e criação de índices concluída. ---")

# --- 2. Configurando o Pipeline RAG com Seleção Dinâmica ---

nomes_das_colecoes = [doc["colecao"] for doc in documentos_para_indexar]
print("\nBases de conhecimento disponíveis:")
for i, nome in enumerate(nomes_das_colecoes, 1):
    print(f"{i}. {nome}")

colecao_escolhida = ""
while colecao_escolhida not in nomes_das_colecoes:
    try:
        escolha = int(input("\nDigite o número da base que você quer consultar: "))
        if 1 <= escolha <= len(nomes_das_colecoes):
            colecao_escolhida = nomes_das_colecoes[escolha - 1]
        else:
            print("Número inválido. Tente novamente.")
    except ValueError:
        print("Entrada inválida. Por favor, digite um número.")

print(f"\nConsultando o índice: {colecao_escolhida}")

qdrant_vector_store = Qdrant.from_existing_collection(
    embedding=embeddings,
    collection_name=colecao_escolhida,
    url=QDRANT_URL,
)

retriever = qdrant_vector_store.as_retriever(search_kwargs={"k": 5})
llm = Ollama(model="llama3")

# ★★★ MUDANÇA AQUI: NOVO PROMPT APRIMORADO ★★★
prompt_template = """
Você é um assistente especializado nas normas e políticas do Programa de Engenharia de Sistemas e Computação (PESC) da Coppe/UFRJ.
Sua principal função é responder perguntas baseando-se estritamente nos documentos internos do programa.

Para responder à pergunta, utilize o seguinte contexto, que foi extraído de um dos documentos oficiais:
- **bolsas_academicas**: Detalhes sobre bolsas CAPES, CNPq, FAPERJ e de projetos.
- **regras_coppe_2020**: Prazos e regras para mestrado e doutorado.
- **roteiro_defesa_remota**: Procedimentos para defesas e qualificações online.

**Contexto Fornecido:**
{context}

**Instruções:**
1. Analise o contexto acima para formular sua resposta.
2. Seja direto e preciso, citando as regras e procedimentos conforme descritos.
3. Se a resposta não puder ser encontrada no contexto fornecido, responda exatamente: 'Com base nos documentos fornecidos, não encontrei uma resposta para isso.'
4. Não utilize nenhum conhecimento externo.

**Pergunta:** {question}
**Resposta Especializada:**
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

# --- 3. Fazendo a Pergunta ---
while True:
    query = input("\nDigite sua pergunta (ou 'sair' para terminar): ")
    if query.lower() == 'sair':
        break

    print("Processando...")
    response = qa_chain.invoke({"query": query})

    print("\n--- Resposta ---")
    print(response['result'])

    print("\n--- Documentos de Origem Encontrados ---")
    for doc in response['source_documents']:
        print(f"Fonte: {doc.metadata.get('source', 'N/A')}, Página: {doc.metadata.get('page', 'N/A')}")
        print(f"Conteúdo: {doc.page_content[:200]}...\n")

print("\n--- Sessão encerrada. ---")