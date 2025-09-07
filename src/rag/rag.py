import os
import sys
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Importa o novo modelo de embeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from logger_config import setup_logger

setup_logger()
logger = logging.getLogger(__name__)
load_dotenv()

# --- Validação das Variáveis de Ambiente ---
required_vars = [
    "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", 
    "AZURE_OPENAI_API_VERSION_EMBEDDING", "EMBEDDING_MODEL_DEPLOYMENT_NAME",
    "QDRANT_URL", "QDRANT_API_KEY"
]
if not all(os.getenv(var) for var in required_vars):
    logger.critical("Uma ou mais variáveis de ambiente essenciais não foram definidas. Verifique seu arquivo .env.")
    sys.exit(1)

# --- Configuração dos Embeddings ---
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("EMBEDDING_MODEL_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION_EMBEDDING"),
)

documentos_para_indexar = [
    {"caminho": "./docs/BolsasAcademicas.txt", "colecao": "bolsas_academicas", "tipo": "txt"},
    {"caminho": "./docs/RegrasCOPPE2020.pdf", "colecao": "regras_coppe_2020", "tipo": "pdf"},
    {"caminho": "./docs/RoteiroDefesaRemota.pdf", "colecao": "roteiro_defesa_remota", "tipo": "pdf"},
]

def criar_indices():
    logger.info("--- Iniciando o processo de indexação com Azure OpenAI Embeddings ---")
    try:
        qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

        for doc_info in documentos_para_indexar:
            caminho, colecao, tipo = doc_info["caminho"], doc_info["colecao"], doc_info["tipo"]
            
            try:
                # Força a recriação para garantir que os embeddings sejam os novos
                logger.info(f"Forçando a recriação da coleção '{colecao}' para atualizar os embeddings.")
                qdrant_client.recreate_collection(
                    collection_name=colecao,
                    vectors_config={"size": 1536, "distance": "Cosine"} # O 'ada-002' usa 1536 dimensões
                )
            except Exception as e:
                logger.error(f"Erro ao recriar a coleção '{colecao}'. Detalhes: {e}")
                continue

            if not os.path.exists(caminho):
                logger.warning(f"Arquivo não encontrado em '{caminho}'. Pulando.")
                continue

            if tipo == "pdf": loader = PyPDFLoader(caminho)
            elif tipo == "txt": loader = TextLoader(caminho)
            else:
                logger.warning(f"Tipo de arquivo não suportado: {caminho}")
                continue

            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            
            logger.info(f"Vetorizando e enviando {len(docs)} chunks para a coleção '{colecao}'...")
            
            Qdrant.from_documents(
                docs, embeddings, 
                url=os.getenv("QDRANT_URL"), 
                api_key=os.getenv("QDRANT_API_KEY"), 
                collection_name=colecao
            )
            logger.info(f"Coleção '{colecao}' indexada com sucesso!")

    except Exception:
        logger.error("Erro fatal durante o processo de indexação.", exc_info=True)
        sys.exit(1)
        
    logger.info("--- Processo de indexação concluído! ---")

if __name__ == "__main__":
    criar_indices()
