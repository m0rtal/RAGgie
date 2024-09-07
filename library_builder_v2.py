import os
import re
import logging
from tqdm import tqdm
from langdetect import detect, LangDetectException
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.pdf import PyPDFium2Loader
from langchain_community.document_loaders.epub import UnstructuredEPubLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from langchain_core.documents import Document

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Модель токенизации
MODEL_NAME = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'

# Инициализация разделителя текста
text_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=100,
    model_name=MODEL_NAME,
    tokens_per_chunk=512
)

# Инициализация модели эмбеддингов
embeddings_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# Инициализация Chroma векторного хранилища
vector_store = Chroma(
    collection_name="collection",
    embedding_function=embeddings_model,
    persist_directory="./chroma_persist"
)


# Функция для очистки текста
def clean_text(text):
    # Удаление ненужных символов и пробелов
    text = re.sub(r'\.{3,}', ' ', text)  # Удаление длинных последовательностей точек
    text = re.sub(r'_ {1,}', ' ', text)  # Удаление длинных последовательностей подчёркиваний
    text = re.sub(r'_{2,}', ' ', text)  # Удаление длинных последовательностей подчёркиваний
    text = re.sub(r'\s+', ' ', text)  # Удаление лишних пробелов
    text = text.strip()  # Удаление пробелов в начале и конце строки
    return text


# Функция для проверки языка текста
def is_valid_language(text):
    try:
        lang = detect(text)
        return lang in ['ru', 'en']
    except LangDetectException:
        return False


# Функция для проверки количества чисел в чанке
def is_valid_chunk(text):
    words = text.split()
    num_words = len(words)
    num_numbers = len(re.findall(r'\d+', text))
    return num_numbers / num_words <= 0.1


# Функция для обработки текста и добавления документов в векторное хранилище
def process_text(file_path, content):
    logging.info(f"Processing file: {file_path}")
    cleaned_content = clean_text(content)
    if cleaned_content:  # Проверка на пустой текст
        chunks = text_splitter.split_text(cleaned_content)
        documents = []
        for chunk in chunks:
            if is_valid_language(chunk) and is_valid_chunk(chunk):
                documents.append(Document(page_content=chunk, metadata={"file_path": file_path}))
        if documents:
            logging.info(f"Adding {len(documents)} documents from {file_path} to the vector store.")
            vector_store.add_documents(documents=documents)
            logging.info(f"Added {len(documents)} documents from {file_path} to the vector store.")
        else:
            logging.info(f"No valid documents found in {file_path}.")
    else:
        logging.info(f"No content found in {file_path}.")


# Функция для загрузки и обработки файла
def process_file(file_path, loader):
    logging.info(f"Loading file: {file_path}")

    try:
        # Проверка, был ли файл уже обработан
        existing_docs = vector_store.similarity_search(query="", filter={"file_path": file_path}, k=1)
        if existing_docs:
            logging.info(f"File {file_path} has already been processed. Skipping.")
            return

        pages = loader.load()
        combined_content = " ".join([page.page_content for page in pages])
        process_text(file_path, combined_content)

    except UnicodeDecodeError as e:
        logging.error(f"UnicodeDecodeError: {e} for file {file_path}")
    except RuntimeError as e:
        logging.error(f"RuntimeError: {e} for file {file_path}")
    except Exception as e:
        logging.error(f"Unexpected error: {e} for file {file_path}")


# Рекурсивный обход директории и сбор файлов
def collect_files(directory):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".pdf", ".epub", ".txt")):
                file_list.append(os.path.join(root, file))
    return file_list


# Загрузка и обработка файлов из папки "Library"
def main():
    library_directory = "/mnt/nfs/Learning"
    file_list = collect_files(library_directory)

    for file_path in tqdm(file_list, desc="Processing files"):
        if file_path.lower().endswith(".pdf"):
            process_file(file_path, PyPDFium2Loader(file_path))
        elif file_path.lower().endswith(".epub"):
            process_file(file_path, UnstructuredEPubLoader(file_path, mode="single", strategy="fast"))
        elif file_path.lower().endswith(".txt"):
            process_file(file_path, TextLoader(file_path, autodetect_encoding=True))
        logging.info(f"Processed and added {file_path} to the vector store.")


if __name__ == "__main__":
    main()

# # Пример поиска по векторному хранилищу
# query = "What are the best workout tips?"
# logging.info(f"Searching for query: {query}")
# results = vector_store.similarity_search(query=query, k=1)
# for doc in results:
#     print(f"* {doc.page_content} [{doc.metadata}]")
