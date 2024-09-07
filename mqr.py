from langchain_core.prompts import PromptTemplate
from library_builder_v2 import vector_store
from langchain_ollama import ChatOllama
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

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

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate up to five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

# Функция для форматирования документов
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Создание MultiQueryRetriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    # prompt=QUERY_PROMPT,
    retriever=vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3}), llm=llm
)

# Создание цепочки обработки запросов и ответов
qa_chain = (
    {
        "context": multi_query_retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Пример запроса
response = qa_chain.invoke("Test query")
print(response)
