from library_builder_v2 import vector_store
from langchain_ollama import ChatOllama
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.globals import set_debug
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

set_debug(True)

# Инициализация модели Ollama
llm = ChatOllama(
    model="llama3.1",
    temperature=0.0,
    # num_predict=256,
)

# Загрузка шаблона для RAG (Retrieval-Augmented Generation)
prompt = hub.pull("rlm/rag-prompt")

# Функция для форматирования документов
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Создание MultiQueryRetriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(k=10), llm=llm
)

# Контекстуализация вопросов
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, multi_query_retriever, contextualize_q_prompt
)

# Создание цепочки для ответа на вопросы
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Управление историей чата
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Цикл для получения запросов от пользователя
session_id = "abc123"
while True:
    user_input = input("Введите ваш вопрос (или 'exit' для выхода): ")
    if user_input.lower() == "exit":
        break

    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )["answer"]
    print("Ответ:", response)
