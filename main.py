# main.py
"""Точка входа для инициализации базы данных.

Key guarantees:
- Централизованная настройка логирования с конфигурацией из .env
- Graceful shutdown при прерывании с корректным закрытием ресурсов
- Четкие коды возврата для CI/CD
- Разделение ответственности (entry point vs business logic)
- Обработка сигналов для graceful termination
"""

import asyncio
import logging
import signal
import sys
from typing import Optional

import asyncpg

from config import app_config, LoggingConfig
from db_initializer import initialize_database


# ======================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ======================================

def setup_logging(config: LoggingConfig) -> None:
    """Настраивает логирование для консольного и файлового вывода."""
    # Создаем директорию для логов, если её нет
    config.file_path.parent.mkdir(parents=True, exist_ok=True)

    handlers = []

    # Файловый хендлер
    file_handler = logging.FileHandler(
        filename=str(config.file_path),
        mode='a',
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    handlers.append(file_handler)

    # Консольный хендлер (если включен)
    if config.console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        handlers.append(console_handler)

    # Базовая настройка логирования
    logging.basicConfig(
        level=getattr(logging, config.level),
        handlers=handlers,
        force=True
    )

    # Настройка уровней для сторонних библиотек
    for logger_name, level in config.third_party_levels.items():
        logging.getLogger(logger_name).setLevel(getattr(logging, level))


# ======================================
# GRACEFUL SHUTDOWN
# ======================================

shutdown_event = asyncio.Event()
shutdown_initiated = False


def signal_handler(sig, frame):
    """Обработчик сигналов для graceful shutdown."""
    global shutdown_initiated
    if shutdown_initiated:
        logging.warning("Получен повторный сигнал завершения. Принудительное завершение.")
        sys.exit(130)

    shutdown_initiated = True
    logging.info(f"Получен сигнал {sig}. Инициируем graceful shutdown...")
    shutdown_event.set()


# ======================================
# ТОЧКА ВХОДА
# ======================================

async def main_async() -> int:
    """Асинхронная основная функция с поддержкой graceful shutdown."""
    global shutdown_initiated

    logging.info("🚀 Запуск инициализации базы данных для RAG сервиса")

    try:
        # Запуск инициализации с таймаутом и возможностью отмены
        init_task = asyncio.create_task(initialize_database())

        # Ожидаем завершения инициализации или сигнала завершения
        done, pending = await asyncio.wait(
            [init_task, asyncio.create_task(shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED
        )

        # Если получили сигнал завершения во время работы
        if shutdown_event.is_set() and not init_task.done():
            logging.info("🛑 Отмена инициализации из-за сигнала завершения")
            init_task.cancel()

            try:
                await init_task
            except asyncio.CancelledError:
                logging.info("✅ Инициализация успешно отменена")
                return 130

        # Обработка результатов инициализации
        if not init_task.done():
            return 1  # Неожиданное состояние

        result = init_task.result()

        if not result.is_success():
            error = result.error
            if isinstance(error, ValueError) and "Недопустимое имя объекта" in str(error):
                logging.error("❌ Ошибка валидации конфигурации: проверьте имена в .env файле")
                return 2
            elif isinstance(error, ConnectionRefusedError):
                logging.error("❌ Невозможно подключиться к PostgreSQL: проверьте хост и порт")
                return 1
            elif isinstance(error, asyncpg.exceptions.InvalidPasswordError):
                logging.error("❌ Неверный пароль для подключения к PostgreSQL")
                return 1
            elif isinstance(error, asyncpg.exceptions.CannotConnectNowError):
                logging.error("❌ PostgreSQL недоступен или не готов к подключению")
                return 1
            else:
                logging.error(f"❌ Инициализация завершилась с ошибкой: {str(error)}")
                return 1

        logging.info("✅ Инициализация базы данных успешно завершена")
        return 0

    except asyncio.CancelledError:
        logging.info("✅ Асинхронная задача успешно отменена")
        return 130
    except Exception as e:
        logging.exception(f"❌ Критическая ошибка при запуске: {str(e)}")
        return 1


if __name__ == "__main__":
    """Основная точка входа для инициализации базы данных.

    Workflow:
    1. Настройка обработчиков сигналов
    2. Настройка логирования из конфигурации
    3. Запуск асинхронной инициализации
    4. Обработка результатов и graceful shutdown

    Exit codes:
        0: Успешное завершение
        1: Ошибка инициализации базы данных
        2: Ошибка конфигурации
        130: Прерывание пользователем (SIGINT/SIGTERM)
    """
    # Регистрация обработчиков сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Настройка логирования
        setup_logging(app_config.logging)

        # Запуск основного асинхронного цикла
        exit_code = asyncio.run(main_async())
        sys.exit(exit_code)

    except KeyboardInterrupt:
        logging.info("🛑 Процесс инициализации прерван пользователем (KeyboardInterrupt)")
        sys.exit(130)
    except Exception as e:
        logging.exception(f"❌ Критическая ошибка при инициализации: {str(e)}")
        sys.exit(1)