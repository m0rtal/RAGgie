# main.py
"""Точка входа для инициализации базы данных.

Key guarantees:
- Централизованная настройка логирования
- Graceful shutdown при прерывании
- Четкие коды возврата для CI/CD
- Разделение ответственности (entry point vs business logic)
"""

import asyncio
import logging
import sys
from db_initializer import initialize_database


# ======================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ======================================

def setup_logging() -> None:
    """Настраивает логирование для консольного и файлового вывода."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler('db_init.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Настройка уровня логирования для сторонних библиотек
    logging.getLogger('asyncpg').setLevel(logging.WARNING)
    logging.getLogger('dotenv').setLevel(logging.INFO)


# ======================================
# ТОЧКА ВХОДА
# ======================================

if __name__ == "__main__":
    """Основная точка входа для инициализации базы данных.

    Workflow:
    1. Настройка логирования
    2. Запуск асинхронной инициализации
    3. Обработка результатов и кодов возврата

    Exit codes:
        0: Успешное завершение
        1: Ошибка инициализации базы данных
        2: Ошибка конфигурации
        130: Прерывание пользователем (SIGINT)
    """
    setup_logging()

    try:
        logging.info("🚀 Запуск инициализации базы данных для RAG сервиса")

        # Запуск асинхронной инициализации
        result = asyncio.run(initialize_database())

        if not result.is_success():
            error = result.error
            if isinstance(error, ValueError) and "Недопустимое имя объекта" in str(error):
                logging.error("❌ Ошибка валидации конфигурации: проверьте имена в .env файле")
                sys.exit(2)
            elif isinstance(error, ConnectionRefusedError):
                logging.error("❌ Невозможно подключиться к PostgreSQL: проверьте хост и порт")
                sys.exit(1)
            else:
                logging.error(f"❌ Инициализация завершилась с ошибкой: {str(error)}")
                sys.exit(1)

        logging.info("✅ Инициализация базы данных успешно завершена")
        sys.exit(0)

    except KeyboardInterrupt:
        logging.info("🛑 Процесс инициализации прерван пользователем")
        sys.exit(130)
    except Exception as e:
        logging.exception(f"❌ Критическая ошибка при запуске: {str(e)}")
        sys.exit(1)