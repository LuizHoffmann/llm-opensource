import os
import sys
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient

# Importa e executa a configuração do logger
from logger_config import setup_logger
setup_logger()

# Cria um logger específico para este módulo
logger = logging.getLogger(__name__)

# --- Configurações Iniciais ---
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    logger.critical("As variáveis de ambiente QDRANT_URL e QDRANT_API_KEY não foram definidas.")
    sys.exit(1)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
documentos_para_indexar = [
    {"caminho": "./docs/BolsasAcademicas.txt", "colecao": "bolsas_academicas", "tipo": "txt"},
    {"caminho": "./docs/RegrasCOPPE2020.pdf", "colecao": "regras_coppe_2020", "tipo": "pdf"},
    {"caminho": "./docs/RoteiroDefesaRemota.pdf", "colecao": "roteiro_defesa_remota", "tipo": "pdf"},
]

def criar_indices():
    logger.info("--- Iniciando o processo de indexação ---")
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        for doc_info in documentos_para_indexar:
            caminho, colecao, tipo = doc_info["caminho"], doc_info["colecao"], doc_info["tipo"]
            
            try:
                qdrant_client.get_collection(collection_name=colecao)
                logger.info(f"A coleção '{colecao}' já existe. Pulando a criação.")
                continue
            except Exception:
                logger.info(f"A coleção '{colecao}' não existe. Criando agora...")

            if not os.path.exists(caminho):
                logger.warning(f"Arquivo não encontrado em '{caminho}'. Pulando.")
                continue

            if tipo == "pdf": loader = PyPDFLoader(caminho)
            elif tipo == "txt": loader = TextLoader(caminho)
            else:
                logger.warning(f"Tipo de arquivo não suportado: {caminho}")
                continue

            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = text_splitter.split_documents(documents)
            
            logger.info(f"Vetorizando e enviando {len(docs)} chunks para a coleção '{colecao}'...")
            
            Qdrant.from_documents(
                docs, embeddings, url=QDRANT_URL, api_key=QDRANT_API_KEY, collection_name=colecao
            )
            logger.info(f"Coleção '{colecao}' criada e indexada com sucesso!")

    except Exception as e:
        logger.error(f"Não foi possível conectar ou processar no Qdrant Cloud.", exc_info=True)
        sys.exit(1)
        
    logger.info("--- Processo de indexação concluído! ---")

if __name__ == "__main__":
    criar_indices()