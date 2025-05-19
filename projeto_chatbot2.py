# projeto_chatbot2.py

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions
from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- Configuração de Arquivos ---
# Obtém o diretório onde o script projeto_chatbot2.py está sendo executado.
# Isso é crucial para que ele encontre os arquivos .env e CSV,
# independentemente de onde o script index.py esteja sendo executado.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, 'base_empresa.csv')
ENV_FILE = os.path.join(SCRIPT_DIR, '.env')

# --- Funções de Inicialização ---
# Estas funções carregam os recursos necessários ou levantam exceções em caso de erro.

def load_api_key():
    """Carrega a chave da API do arquivo .env."""
    if not os.path.exists(ENV_FILE):
        # Levanta uma exceção que o script que chama (index.py) pode capturar
        raise FileNotFoundError(f"ERRO: O arquivo .env não foi encontrado em {ENV_FILE}")

    load_dotenv(ENV_FILE)
    api_key = os.environ.get('GOOGLE_API_KEY')

    if not api_key:
        raise ValueError('ERRO: A chave GOOGLE_API_KEY não foi encontrada nas variáveis de ambiente.')

    return api_key

def load_data():
    """Carrega dados do arquivo CSV em objetos Document."""
    documentos = []
    if not os.path.exists(CSV_FILE):
         raise FileNotFoundError(f"ERRO: O arquivo CSV da base de dados não foi encontrado em {CSV_FILE}")

    try:
        loader = CSVLoader(
            file_path=CSV_FILE,
            csv_args={'delimiter': ';'},
            encoding='utf-8'
        )
        documentos = loader.load()
    except Exception as e:
        raise IOError(f"ERRO: Não foi possível ler ou processar o arquivo CSV '{CSV_FILE}'. Verifique o formato (delimitador ';'), codificação (UTF-8) e conteúdo. Detalhes: {e}")

    if not documentos:
         # Retorna lista vazia, o chamador (index.py) decidirá como lidar
         print(f"\nAVISO (projeto_chatbot2.py): Nenhum dado foi carregado do arquivo CSV '{CSV_FILE}'.")

    return documentos

def initialize_retriever(documents, api_key):
    """Inicializa embeddings e constrói o vetorstore retriever."""
    if not documents:
         # Se não há documentos, não inicializa o retriever e retorna None
         print("AVISO (projeto_chatbot2.py): Não há documentos para criar o vetorstore.")
         return None # Retorna None para indicar que o retriever não pôde ser criado

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        return retriever
    except Exception as e:
         raise RuntimeError(f"ERRO: Não foi possível inicializar embeddings ou vetorstore. Detalhes: {e}")


def initialize_gemini_chat(api_key):
    """Inicializa a sessão de chat com o modelo Gemini."""
    tipo = 'gemini-1.5-flash'
    chat_config = types.GenerateContentConfig(
        system_instruction= "Você é um assistente pessoal da empresa, especializado em fornecer informações com base nos dados que lhe são apresentados. Responda de forma clara, objetiva e amigável. Se a informação solicitada não estiver no contexto fornecido, diga educadamente que não possui essa informação. Caso não tenha informações na base, e através de pesquisas consiga montar uma resposta que não fuja muito da pergunta."
    )
    try:
        client = genai.Client(api_key=api_key)
        chat = client.chats.create(model=tipo, config=chat_config)
        return chat
    except google_exceptions.GoogleAPIError as e:
        raise ConnectionError(f"ERRO de API do Google: Não foi possível conectar ou inicializar o modelo Gemini '{tipo}'. Verifique sua API key, conexão e disponibilidade do modelo. Detalhes: {e}")
    except Exception as e:
         raise RuntimeError(f"ERRO: Não foi possível inicializar o chat Gemini. Detalhes: {e}")

# --- Função Principal de Resposta do Chatbot ---

def get_bot_response(question, retriever, chat_session):
    """
    Obtém a resposta do modelo Gemini com base na pergunta e contexto.

    Args:
        question (str): A pergunta do usuário.
        retriever: O objeto retriever da base de dados (ou None).
        chat_session: A sessão de chat do Gemini (ou None).

    Returns:
        str: A resposta do chatbot ou uma mensagem de erro.
    """
    if not chat_session:
         # Retorna uma mensagem de erro se a sessão de chat não foi inicializada
         return "Erro interno: A sessão do chat não foi inicializada corretamente."

    contexto_formatado = ""
    if retriever: # Verifica se o retriever foi inicializado com sucesso
        try:
            # Com a pergunta do usuário, o retriever busca documentos relevantes
            documentos_relevantes: list[Document] = retriever.invoke(question)
            # Formata o texto dos documentos relevantes para o prompt do modelo
            contexto_formatado = "\n---\n".join([doc.page_content for doc in documentos_relevantes])
        except Exception as e:
             # Loga o erro, mas permite que a resposta do Gemini prossiga (sem contexto do CSV)
             print(f"AVISO (projeto_chatbot2.py): Ocorreu um erro ao buscar documentos relevantes: {e}")
             contexto_formatado = "Não foi possível buscar contexto na base de dados devido a um erro."

    # Define o prompt final que será enviado ao modelo Gemini, incluindo o contexto (se houver).
    prompt_final = f"""
    Contexto de informações da base de dados:
    {contexto_formatado if contexto_formatado else "Nenhum contexto da base de dados encontrado."}

    ---

    Pergunta do usuário: {question}

    ---

    Com base no contexto fornecido acima (se houver) e na pergunta do usuário, responda de forma útil e concisa.
    - Se a resposta puder ser encontrada no contexto, use as informações fornecidas.
    - Se a resposta não estiver no contexto, mas você tiver conhecimento geral relevante, utilize-o para formar a resposta.
    - Se a resposta não estiver no contexto e você também não tiver conhecimento generalizado sobre o assunto, informe que a informação específica não foi encontrada na base de dados.
    Mantenha a persona de assistente pessoal da empresa.
    """

    try:
        # Envia o prompt para o modelo e obtém a resposta
        resposta = chat_session.send_message(prompt_final)
        return resposta.text # Retorna apenas o texto da resposta
    except Exception as e:
        # Retorna a mensagem de erro para ser exibida na interface do usuário
        return f"Ocorreu um erro durante a interação com o modelo de chat: {e}"

# Este arquivo não tem um bloco if __name__ == "__main__":
# porque ele não é feito para ser executado diretamente, apenas importado.