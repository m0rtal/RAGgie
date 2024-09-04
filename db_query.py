import itertools
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from textwrap import fill
from sentence_transformers import CrossEncoder

# Инициализация клиента Chroma
def initialize_chroma_client(path="chroma_db"):
    return chromadb.PersistentClient(path=path)

# Инициализация функции встраивания
def initialize_embedding_function(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"):
    return SentenceTransformerEmbeddingFunction(model_name=model_name)

# Получение или создание коллекции
def get_or_create_collection(chroma_client, collection_name, embedding_function):
    return chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

# Выполнение запроса
def query_collection(chroma_collection, query, n_results):
    results = chroma_collection.query(query_texts=query, n_results=n_results)
    flat_documents = list(itertools.chain.from_iterable(results['documents']))
    return flat_documents

# Форматирование текста
def format_text(text, width=160):
    return fill(text, width=width)

# Печать результатов
def print_results(retrieved_documents):
    print("\nTop retrieved documents:\n")
    for i, document in enumerate(retrieved_documents):
        print(f"Document {i+1}:\n")
        print(format_text(document))
        print('\n' + '-'*120 + '\n')

# Основная функция для выполнения запроса
def perform_query(query, collection_name="documents", path="chroma_db", model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1", n_results=10):
    chroma_client = initialize_chroma_client(path=path)
    embedding_function = initialize_embedding_function(model_name=model_name)
    chroma_collection = get_or_create_collection(chroma_client, collection_name, embedding_function)
    retrieved_documents = query_collection(chroma_collection, query, n_results=n_results)
    unique_documents = list(set(retrieved_documents))
    return unique_documents

# Функция для ранжирования документов
def rank_documents(query, documents, cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2", return_num=10):
    cross_encoder = CrossEncoder(cross_encoder_model)

    # Ранжирование документов по запросу
    pairs = [[query, doc] for doc in documents]
    scores = cross_encoder.predict(pairs)

    # Объединение результатов и их оценок
    combined_scores = []
    for i, doc in enumerate(documents):
        combined_scores.append((scores[i], doc))

    # Сортировка по оценкам и выбор топ-10 результатов
    combined_scores.sort(reverse=True, key=lambda x: x[0])
    top_documents = [doc for score, doc in combined_scores[:return_num]]

    return top_documents
