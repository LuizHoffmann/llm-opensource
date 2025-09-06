import os
import sys
import logging
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
# CORREÇÃO: A importação correta do Ollama é feita a partir do langchain_community
from langchain_community.llms import Ollama
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient

# Importa e executa a configuração do logger
from logger_config import setup_logger
setup_logger()
logger = logging.getLogger(__name__)

# --- Carregamento de Configurações e Modelos (com Cache) ---
load_dotenv()

@st.cache_resource
def get_qdrant_client():
    """ Carrega o cliente Qdrant uma vez e o mantém em cache. """
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        st.error("As variáveis de ambiente do Qdrant não foram definidas. Verifique seu arquivo .env.")
        st.stop()
    return QdrantClient(url=url, api_key=api_key)

@st.cache_resource
def get_embeddings_model():
    """ Carrega o modelo de embeddings uma vez e o mantém em cache. """
    logger.info("Carregando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    logger.info("Modelo de embeddings carregado.")
    return embeddings

# --- Configuração do Pipeline RAG ---

def setup_qa_chain(collection_name):
    """ Configura e retorna a cadeia de RetrievalQA para uma coleção específica. """
    logger.info(f"Configurando a cadeia RAG para a coleção: {collection_name}")
    client = get_qdrant_client()
    embeddings = get_embeddings_model()
    
    qdrant_vector_store = Qdrant(
        client=client, 
        collection_name=collection_name, 
        embeddings=embeddings
    )
    
    retriever = qdrant_vector_store.as_retriever(search_kwargs={"k": 5})
    llm = Ollama(model="llama3")
    
    prompt_template = """
    Você é um assistente especializado nas normas e políticas do Programa de Engenharia de Sistemas e Computação (PESC) da Coppe/UFRJ.
    Sua principal função é responder perguntas baseando-se estritamente nos documentos internos do programa.
    **Contexto Fornecido:**
    {context}
    **Instruções:**
    1. Analise o contexto acima para formular sua resposta.
    2. Se a resposta não puder ser encontrada no contexto, responda exatamente: 'Com base nos documentos fornecidos, não encontrei uma resposta para isso.'
    3. Não utilize nenhum conhecimento externo.
    **Pergunta:** {question}
    **Resposta Especializada:**
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True, 
        chain_type_kwargs={"prompt": PROMPT}
    )
    logger.info("Cadeia RAG configurada com sucesso.")
    return qa_chain

# --- Interface Gráfica (Streamlit) ---

st.set_page_config(page_title="Assistente PESC/Coppe", page_icon="🤖")
st.title("🤖 Assistente de Normas do PESC/Coppe")

# Mapeamento de nomes amigáveis para nomes das coleções
TOPICS = {
    "Bolsas acadêmicas": "bolsas_academicas",
    "Regras da Coppe": "regras_coppe_2020",
    "Roteiro para defesa": "roteiro_defesa_remota",
}

# Inicialização do estado da sessão
if "conversation_topic" not in st.session_state:
    st.session_state.conversation_topic = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# --- Lógica da Aplicação ---

# 1. Se nenhum tópico foi escolhido, mostra a seleção
if st.session_state.conversation_topic is None:
    st.info("Olá! Sou seu assistente para assuntos acadêmicos do PESC. Sobre qual tópico você gostaria de conversar?")
    
    selected_topic_friendly_name = st.selectbox(
        "Escolha um assunto:",
        options=list(TOPICS.keys()),
        index=None,
        placeholder="Selecione um tópico..."
    )

    if selected_topic_friendly_name:
        collection_name = TOPICS[selected_topic_friendly_name]
        st.session_state.conversation_topic = collection_name
        
        # Configura a cadeia RAG e a armazena na sessão
        with st.spinner(f"Preparando o assistente para falar sobre '{selected_topic_friendly_name}'..."):
            st.session_state.qa_chain = setup_qa_chain(collection_name)
        
        # Adiciona mensagem inicial do assistente
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Pronto! Pode me perguntar qualquer coisa sobre '{selected_topic_friendly_name}'.",
            "sources": [] # Mensagem inicial não tem fontes
        })
        st.rerun()

# 2. Se um tópico já foi escolhido, mostra a interface de chat
else:
    # Exibe as mensagens do histórico
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Adiciona o botão de fontes para respostas do assistente que tenham fontes
            if message["role"] == "assistant" and message["sources"]:
                with st.expander("Ver fontes da resposta"):
                    for doc in message["sources"]:
                        st.info(
                            f"**Fonte:** `{doc.metadata.get('source', 'N/A')}` | "
                            f"**Página:** `{doc.metadata.get('page', 'N/A')}`\n\n"
                            f"**Conteúdo:**\n\n---\n\n{doc.page_content}"
                        )
    
    # Captura a nova pergunta do usuário
    if prompt := st.chat_input("Faça sua pergunta..."):
        # Adiciona a pergunta do usuário ao histórico e exibe na tela
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Processa a pergunta e obtém a resposta do assistente
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                logger.info(f"Invocando a cadeia RAG com a query: '{prompt}'")
                response = st.session_state.qa_chain.invoke({"query": prompt})
                answer = response['result']
                source_documents = response['source_documents']
                
                st.markdown(answer)
                
                # Adiciona o botão de fontes para a nova resposta
                with st.expander("Ver fontes da resposta"):
                    for doc in source_documents:
                        st.info(
                            f"**Fonte:** `{doc.metadata.get('source', 'N/A')}` | "
                            f"**Página:** `{doc.metadata.get('page', 'N/A')}`\n\n"
                            f"**Conteúdo:**\n\n---\n\n{doc.page_content}"
                        )
        
        # Adiciona a resposta completa do assistente (com fontes) ao histórico
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "sources": source_documents
        })

