import os
import logging
from dotenv import load_dotenv
import streamlit as st
from src.utils.logger_config import setup_logger

# Importa a função que cria nosso agente
from src.llm.agent import get_agent

# --- Configurações Iniciais ---
setup_logger()
logger = logging.getLogger(__name__)
load_dotenv()

# --- Carregamento do Agente com Cache ---
@st.cache_resource
def load_agent():
    """Carrega o agente LangGraph uma vez e o mantém em cache."""
    logger.info("Carregando e compilando o agente LangGraph com roteador...")
    return get_agent()

# --- Interface Gráfica (Streamlit) ---
st.set_page_config(page_title="Assistente PESC/Coppe", page_icon="🤖")
st.title("🤖 Assistente de Normas do PESC/Coppe")

# Carrega o agente
agent = load_agent()

# Inicialização do estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Olá! Sou seu assistente para assuntos acadêmicos do PESC. Como posso ajudar?",
        "sources": []
    }]

# --- Lógica da Aplicação ---

# Exibe o histórico da conversa
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Ver fontes da resposta"):
                for doc in message["sources"]:
                    st.info(f"**Fonte:** `{doc.metadata.get('source', 'N/A')}` | **Página:** `{doc.metadata.get('page', 'N/A')}`\n\n---\n\n{doc.page_content}")

# Captura a pergunta do usuário
if prompt := st.chat_input("Faça sua pergunta sobre bolsas, regras ou defesa..."):
    # Adiciona a pergunta do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Exibe a resposta do assistente
    with st.chat_message("assistant"):
        with st.spinner("Analisando sua pergunta e buscando nos documentos..."):
            # O estado inicial agora só precisa da pergunta
            initial_state = {"question": prompt}
            
            # Invoca o agente para obter a resposta completa
            final_state = agent.invoke(initial_state)
            
            st.markdown(final_state["answer"])
            
            # Mostra as fontes
            with st.expander("Ver fontes da resposta"):
                for doc in final_state["documents"]:
                     st.info(f"**Fonte:** `{doc.metadata.get('source', 'N/A')}` | **Página:** `{doc.metadata.get('page', 'N/A')}`\n\n---\n\n{doc.page_content}")

    # Adiciona a resposta completa do agente ao histórico
    st.session_state.messages.append({
        "role": "assistant", 
        "content": final_state["answer"], 
        "sources": final_state["documents"]
    })