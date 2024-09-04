from langchain_ollama import ChatOllama
from db_query import perform_query, print_results, rank_documents

def translate_query(llm, query, target_language):
    messages = [
        (
            "system",
            f"You are a helpful assistant that translates Russian to {target_language}. Translate the user sentence. "
            "Output only translated sentence without any additional information. ",
        ),
        ("human", query),
    ]
    return llm.invoke(messages).content

def generate_additional_queries(llm, query, language):
    messages = [
        (
            "system",
            f"You are a helpful expert research assistant. Your users are asking different questions about some topic."
            f"Suggest up to ten additional related questions to help them find the information they need, for the provided question. "
            f"Regardless of a query language, always answer only in {language} language. "
            f"Do not use 'it' in sentences, always mention the topic/subject in full. "
            f"Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic."
            f"Make sure they are complete questions, and that they are related to the original question."
            f"Output one question per line. Do not number the questions."
        ),
        ("human", query),
    ]
    ai_msg = llm.invoke(messages)
    return ai_msg.content.split('\n')

def main(original_query):
    llm = ChatOllama(
        model="llama3.1",
        temperature=0,
        # other params...
    )

    print(original_query)

    # Перевод оригинального запроса
    translated_query = translate_query(llm, original_query, "English")
    print(translated_query)

    # Генерация дополнительных запросов на английском
    additional_english_queries = generate_additional_queries(llm, translated_query, "English")
    print(*additional_english_queries, sep='\n')

    # Генерация дополнительных запросов на русском
    additional_russian_queries = generate_additional_queries(llm, original_query, "Russian")
    print(*additional_russian_queries, sep='\n')

    # Выполнение англоязычных запросов и сохранение результатов
    english_queries = [translated_query] + additional_english_queries
    english_results = perform_query(english_queries, n_results=10)
    english_ranked_results = rank_documents(translated_query, english_results, return_num=5)

    # Выполнение русскоязычных запросов и добавление результатов
    russian_queries = [original_query] + additional_russian_queries
    russian_results = perform_query(russian_queries, n_results=10)
    russian_ranked_results = rank_documents(original_query, russian_results, return_num=5)

    # Вывод результатов
    all_results = english_ranked_results + russian_ranked_results
    print_results(list(set(all_results)))

if __name__ == "__main__":
    # main(original_query="Хочу создать своё онлайн-сообщество, посвящённое выживальщикам. С чего лучше начать?")
    main(original_query="query?")
