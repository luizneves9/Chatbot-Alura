# index.py

import streamlit as st
import os
# Importa as funções do seu arquivo de lógica
from projeto_chatbot2 import (
    load_api_key,
    load_data,
    initialize_retriever,
    initialize_gemini_chat,
    get_bot_response # Importa a função que gera a resposta
)

# --- Layout da Aplicação Streamlit ---

st.title("Chatbot Alura - IA")
st.write("Chatbot com informações da empresa para fins de conhecimentos.")

# --- Inicialização e Carregamento de Recursos com Cache ---
# Usamos as funções importadas, MAS as envolvemos com os decoradores de cache do Streamlit.
# Isso garante que o carregamento da API key, dados e a inicialização dos modelos/stores
# aconteçam apenas uma vez, mesmo que o script index.py seja re-executado
# em cada interação do usuário (que é como o Streamlit funciona).

@st.cache_resource(show_spinner="Carregando Configurações...")
def load_config():
    """
    Tenta carregar a API Key de variáveis de ambiente (Secrets do Streamlit Cloud)
    ou de um arquivo .env localmente.
    """
    # Primeiro, tenta ler a variável de ambiente diretamente.
    # No Streamlit Cloud, esta variável será populada se você configurou os Secrets.
    # Localmente, pode já estar definida no seu ambiente ou será None inicialmente.
    api_key = os.environ.get('GOOGLE_API_KEY')

    # Se a variável de ambiente NÃO estiver definida,
    # tenta carregar do arquivo .env (útil para desenvolvimento local).
    if not api_key:
        env_path = ENV_FILE # Usa o caminho definido globalmente no index.py
        if os.path.exists(env_path):
            print(f"Tentando carregar .env localmente de {env_path}") # Opcional: para debug local
            load_dotenv(env_path)
            # Tenta ler a variável novamente após carregar o .env
            api_key = os.environ.get('GOOGLE_API_KEY')
        else:
            # Este 'else' será executado no Streamlit Cloud onde .env não existe
            # E também localmente se .env não existe E a variável não está no ambiente
            print(f"Aviso: Arquivo .env não encontrado em {env_path}.") # Opcional: para debug

    # Se após todas as tentativas a chave ainda não foi encontrada, é um erro crítico.
    if not api_key:
        st.error('ERRO: A chave GOOGLE_API_KEY não foi encontrada.')
        st.error('Certifique-se de que GOOGLE_API_KEY está definida:')
        st.error('- Nos Secrets do Streamlit Cloud (para deploy)')
        st.error('- OU no arquivo .env (para desenvolvimento local)')
        st.stop() # Para a execução do Streamlit

    return api_key

@st.cache_data(show_spinner="Carregando Base de Dados CSV...")
def get_cached_documents():
    """Carrega os documentos usando a função do projeto_chatbot2.py e aplica cache."""
    try:
        return load_data()
    except Exception as e:
        st.error(f"Erro de Inicialização (Base CSV): {e}")
        st.stop() # Para a execução do Streamlit se o CSV falhar
    return None # Não deve chegar aqui

@st.cache_resource(show_spinner="Inicializando Vetorstore...")
def get_cached_retriever(_documents, _api_key):
    """Inicializa o retriever usando a função do projeto_chatbot2.py e aplica cache."""
    # Passamos os objetos para o cache (_documents, _api_key).
    # O Streamlit os usará para determinar se precisa re-executar esta função.
    try:
        return initialize_retriever(_documents, _api_key)
    except Exception as e:
        st.error(f"Erro de Inicialização (Vetorstore): {e}")
        # Não paramos o app aqui, pois o bot ainda pode dar respostas gerais
        return None # Retorna None para indicar que o retriever não foi inicializado

@st.cache_resource(show_spinner="Inicializando Modelo Gemini...")
def get_cached_chat_session(_api_key):
    """Inicializa a sessão de chat Gemini usando a função do projeto_chatbot2.py e aplica cache."""
    try:
        return initialize_gemini_chat(_api_key)
    except Exception as e:
        st.error(f"Erro de Inicialização (Modelo Gemini): {e}")
        st.stop() # Paramos o app aqui, pois o chat é essencial
    return None # Não deve chegar aqui

# --- Carregar os Recursos para a Sessão ---
# Estas chamadas disparam as funções cacheadas acima.
# Elas serão executadas rapidamente após a primeira vez.
api_key = load_config() # Carrega a API Key
documents = get_cached_documents() # Carrega os dados do CSV
retriever = get_cached_retriever(documents, api_key) # Inicializa o retriever (passa os dados e a key)
chat_session = get_cached_chat_session(api_key) # Inicializa o chat Gemini (passa a key)


# --- Gerenciamento do Histórico do Chat (Streamlit Session State) ---
# st.session_state permite que variáveis persistam entre as re-execuções do script.
if 'messages' not in st.session_state:
    st.session_state.messages = [] # Inicializa o histórico como uma lista vazia
    # Adiciona uma mensagem inicial do assistente se o chat foi inicializado
    if chat_session:
         st.session_state.messages.append({"role": "assistant", "content": "Olá! Como posso ajudar hoje?"})
    else:
         st.session_state.messages.append({"role": "assistant", "content": "Erro ao iniciar o chatbot. Por favor, verifique as mensagens de erro acima."})


# --- Exibir Histórico de Mensagens ---
# Itera sobre as mensagens armazenadas na session_state e as exibe na interface.
for message in st.session_state.messages:
    # st.chat_message formata a mensagem com um ícone e estilo de balão de chat
    with st.chat_message(message["role"]): # 'user' ou 'assistant'
        st.markdown(message["content"]) # Exibe o texto da mensagem


# --- Entrada de Texto para o Usuário (Barra de Chat) ---
# st.chat_input cria a barra de entrada na parte inferior.
# 'prompt := st.chat_input(...)' atribui o texto digitado à variável 'prompt'
# quando o usuário envia a mensagem (e é True), ou None (e é False) caso contrário.
# A barra de chat só é mostrada se a sessão de chat foi inicializada com sucesso.
if chat_session:
    if prompt := st.chat_input("Digite sua pergunta aqui..."):

        # --- Processar a Pergunta do Usuário ---
        # 1. Adiciona a pergunta do usuário ao histórico na session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        # 2. Exibe a pergunta do usuário imediatamente na interface
        with st.chat_message("user"):
            st.markdown(prompt)

        # 3. Obtém a resposta do chatbot
        # Chama a função importada de projeto_chatbot2.py para obter a resposta.
        # Passamos a pergunta do usuário e os recursos (retriever, chat_session)
        # que foram carregados/cacheados anteriormente.
        response_text = get_bot_response(prompt, retriever, chat_session)

        # 4. Adiciona a resposta do assistente ao histórico na session state
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        # 5. Exibe a resposta do assistente na interface
        with st.chat_message("assistant"):
            st.markdown(response_text)

        # st.chat_input lida com a re-execução do script automaticamente após o envio.
        # A próxima execução exibirá todo o histórico atualizado devido ao loop acima.

else:
    # Mensagem para o usuário se a inicialização do chat falhou
    st.warning("O chatbot não está pronto. Por favor, corrija os erros de inicialização exibidos acima.")


# Em scripts Streamlit, não usamos um bloco main() e if __name__ == "__main__":
# O script é executado de cima para baixo a cada interação.
