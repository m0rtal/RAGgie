from library_builder_v2 import vector_store
from langchain_ollama import ChatOllama
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.globals import set_debug

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

# Создание цепочки обработки запросов и ответов
qa_chain = (
    {
        "context": vector_store.as_retriever(k=10) | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Пример запроса
response = qa_chain.invoke("Test query")
print(response)
