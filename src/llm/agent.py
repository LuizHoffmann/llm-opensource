import os
import logging
from typing import List, TypedDict
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_qdrant import Qdrant
from langgraph.graph import StateGraph, END

# --- Configuração Inicial ---
# Garante que as variáveis de ambiente sejam carregadas neste módulo também
load_dotenv()
logger = logging.getLogger(__name__)

# --- Validação das Variáveis de Ambiente ---
# Adicionamos esta verificação para garantir que o agente tenha as credenciais
required_vars = [
    "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION_LLM", "EMBEDDING_MODEL_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION_EMBEDDING", "QDRANT_URL", "QDRANT_API_KEY"
]
if not all(os.getenv(var) for var in required_vars):
    # Esta mensagem de erro agora aparecerá nos logs do Streamlit se o .env não for carregado
    error_message = "ERRO CRÍTICO: Uma ou mais variáveis de ambiente essenciais não foram carregadas no agente. Verifique o arquivo .env."
    logger.critical(error_message)
    raise ValueError(error_message)


# --- Definição de Estruturas (Schema) ---
class RouteQuery(BaseModel):
    """Roteia a pergunta de um usuário para a base de conhecimento mais relevante."""
    collection_name: str = Field(
        description="O nome da coleção para onde a pergunta deve ser roteada.",
        enum=["bolsas_academicas", "regras_coppe_2020", "roteiro_defesa_remota"],
    )

# --- Definição do Estado do Agente ---
class AgentState(TypedDict):
    question: str
    collection_name: str
    documents: List[Document]
    answer: str

# --- Funções dos Nós do Grafo ---
def route_question(state: AgentState) -> AgentState:
    """Nó do grafo: Analisa a pergunta e decide para qual coleção rotear."""
    logger.info(f"Roteando a pergunta: '{state['question']}'")
    llm_router = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION_LLM"),
        temperature=0,
    ).with_structured_output(RouteQuery)
    
    prompt = f"""Você é um especialista em rotear perguntas de usuários para a base de dados correta.
    Com base na pergunta do usuário, escolha a coleção mais apropriada.

    Coleções disponíveis:
    - 'bolsas_academicas': Para perguntas sobre auxílios, valores, tipos de bolsa e financiamento.
    - 'regras_coppe_2020': Para perguntas sobre prazos, regras gerais, matrículas e disciplinas.
    - 'roteiro_defesa_remota': Para perguntas sobre o processo de defesa de mestrado ou doutorado, qualificação e procedimentos remotos.

    Pergunta do usuário: "{state['question']}"
    """
    
    route = llm_router.invoke(prompt)
    logger.info(f"Pergunta roteada para a coleção: '{route.collection_name}'")
    return {"collection_name": route.collection_name, **state}

def retrieve_documents(state: AgentState) -> AgentState:
    logger.info(f"Recuperando documentos da coleção '{state['collection_name']}'...")
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("EMBEDDING_MODEL_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION_EMBEDDING"),
    )
    qdrant = Qdrant.from_existing_collection(
        embedding=embeddings,
        collection_name=state["collection_name"],
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    retriever = qdrant.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(state["question"])
    logger.info(f"{len(docs)} documentos recuperados.")
    return {"documents": docs, **state}

def generate_answer(state: AgentState) -> AgentState:
    logger.info("Gerando resposta com o LLM...")
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION_LLM"),
        temperature=0,
    )
    prompt_template = """
    Você é um assistente especializado nas normas do PESC/Coppe/UFRJ.
    Use o contexto abaixo para responder à pergunta.
    Contexto: {context}
    Pergunta: {question}
    Resposta:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = prompt | llm
    context = "\n\n".join([doc.page_content for doc in state["documents"]])
    response = chain.invoke({"context": context, "question": state["question"]})
    logger.info("Resposta gerada.")
    return {"answer": response.content, **state}

# --- Montagem e Compilação do Agente ---
def get_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("router", route_question)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.set_entry_point("router")
    workflow.add_edge("router", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    agent = workflow.compile()
    logger.info("Agente LangGraph com roteador compilado com sucesso.")
    return agent

