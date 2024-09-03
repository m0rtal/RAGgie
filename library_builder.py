import os
import chromadb
from langchain_community.document_loaders.pdf import PyPDFium2Loader
from langchain_community.document_loaders.epub import UnstructuredEPubLoader
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import logging
import ftfy  # Подключаем ftfy для исправления текста
from langdetect import detect, LangDetectException
import re

# Инициализация логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Путь к папке с файлами
folder_path = "/mnt/nfs/Learning"

# https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
transformer_model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# Инициализация клиента Chroma
chroma_client = chromadb.PersistentClient(path="chroma_db")
embedding_function = SentenceTransformerEmbeddingFunction(model_name=transformer_model)

# Создание или получение коллекции
collection_name = "documents"
chroma_collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

# Функция для проверки, был ли файл уже обработан
def is_file_processed(file_id):
    results = chroma_collection.get(ids=[f"{file_id}_0"])
    return len(results['documents']) > 0

# Функция для проверки языка чанка
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

# Функция для обработки одного файла (PDF или EPUB)
def process_file(file_path, collection, file_type):
    try:
        if file_type == 'pdf':
            loader = PyPDFium2Loader(file_path)
        elif file_type == 'epub':
            loader = UnstructuredEPubLoader(file_path, mode="elements", strategy="fast")
        else:
            logger.warning(f"Unsupported file type for {file_path}. Skipping.")
            return

        # Загрузка страниц или элементов
        documents = loader.load()

        # Объединение текста всех страниц в один большой текст
        full_text = ""
        for document in documents:
            full_text += document.page_content + "\n"

        # Исправление текста с помощью ftfy
        full_text = ftfy.fix_text(full_text)

        # Разделение текста на чанки
        token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=512, model_name=transformer_model)
        token_chunks = token_splitter.split_text(full_text)

        # Проверка на пустые токенизированные чанки
        if not token_chunks:
            logger.warning(f"Warning: No token chunks found for file {file_path}.")
            return

        # Фильтрация чанков по языку и количеству чисел
        valid_chunks = [chunk for chunk in token_chunks if is_valid_language(chunk) and is_valid_chunk(chunk)]

        # Проверка на пустые валидные чанки
        if not valid_chunks:
            logger.warning(f"Warning: No valid chunks found for file {file_path}.")
            return

        # Генерация уникального идентификатора для файла
        file_id = os.path.basename(file_path)

        # Проверка, был ли файл уже обработан
        if is_file_processed(file_id):
            logger.info(f"File {file_id} has already been processed.")
            return

        # Генерация идентификаторов для чанков
        ids = [f"{file_id}_{i}" for i in range(len(valid_chunks))]

        # Добавление чанков в коллекцию
        logger.info(f"Adding {len(valid_chunks)} chunks from {file_path} to collection.")
        collection.add(ids=ids, documents=valid_chunks, metadatas=[{"file_id": file_id}] * len(valid_chunks))
        logger.info(f"Added {len(valid_chunks)} chunks from {file_path} to collection.")

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")

# Рекурсивный обход всех файлов в папке и её подкаталогах
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        file_path = os.path.join(root, filename)
        if filename.lower().endswith(".pdf"):
            process_file(file_path, chroma_collection, 'pdf')
        elif filename.lower().endswith(".epub"):
            process_file(file_path, chroma_collection, 'epub')

logger.info("Все файлы обработаны и добавлены в коллекцию.")
