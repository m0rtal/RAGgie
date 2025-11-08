# db_initializer.py
"""Инициализация PostgreSQL базы данных для RAG сервиса.

Key guarantees:
- Идемпотентность всех операций (безопасный повторный запуск)
- Оптимизированная схема только для векторного поиска
- Безопасность: строгая валидация идентификаторов
- Отказоустойчивость: обработка ошибок через Railway-ориентированное программирование
- Полная совместимость с предоставленным config.py

Trade-offs considered:
- Полностью удалён полнотекстовый поиск (tsvector) как избыточный для RAG
- Использованы значения по умолчанию для параметров векторных индексов
- Упрощена логика создания индексов для совместимости с конфигурацией
"""

import logging
import re
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar, cast, Final, Callable
from functools import wraps

import asyncpg
from dotenv import load_dotenv
from config import DBConfig, DBSchemaConfig

# ======================================
# КОНСТАНТЫ И ВАЛИДАЦИЯ
# ======================================

# Безопасные паттерны для валидации идентификаторов
SAFE_IDENTIFIER_PATTERN: Final[str] = r'^[a-zA-Z0-9_]+$'
SAFE_DATABASE_NAME_PATTERN: Final[str] = r'^[a-zA-Z0-9_-]+$'


def validate_identifier(name: str, pattern: str = SAFE_IDENTIFIER_PATTERN) -> str:
    """Валидация идентификаторов для предотвращения SQL инъекций."""
    if not re.match(pattern, name):
        raise ValueError(f"Недопустимое имя объекта: {name}")
    return name


T = TypeVar('T')


class Result(Generic[T]):
    """Railway-oriented programming для обработки ошибок с поддержкой типов."""
    __slots__ = ('_value', '_error')

    def __init__(self, value: Optional[T] = None, error: Optional[Exception] = None):
        self._value = value
        self._error = error

    @staticmethod
    def success(value: T) -> 'Result[T]':
        """Создаёт успешный результат."""
        return Result(value=value)

    @staticmethod
    def failure(error: Exception) -> 'Result[T]':
        """Создаёт неудачный результат с ошибкой."""
        return Result(error=error)

    def is_success(self) -> bool:
        """Проверяет, является ли результат успешным."""
        return self._error is None

    def unwrap(self) -> T:
        """Возвращает значение или выбрасывает исключение при ошибке."""
        if self._error:
            raise self._error
        if self._value is None:
            raise ValueError("No value present in successful Result")
        return cast(T, self._value)

    def map(self, func: Callable[[T], Any]) -> 'Result[Any]':
        """Применяет функцию к значению в случае успеха."""
        if self.is_success():
            try:
                return Result.success(func(self._value))
            except Exception as e:
                return Result.failure(e)
        return Result.failure(self._error)


# ======================================
# ФУНКЦИОНАЛЬНЫЕ КОМПОНЕНТЫ
# ======================================

async def create_db_connection(config: DBConfig) -> Result[asyncpg.Pool]:
    """Создает пул соединений с PostgreSQL с таймаутами и валидацией."""
    try:
        pool = await asyncpg.create_pool(
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.user,
            password=config.password,
            min_size=config.min_connections,
            max_size=config.max_connections,
            timeout=30.0,
            command_timeout=60.0,
            max_inactive_connection_lifetime=300.0
        )
        return Result.success(pool)
    except Exception as e:
        return Result.failure(e)


async def ensure_database_exists(config: DBConfig) -> Result[None]:
    """Проверяет существование базы данных и создаёт её при отсутствии."""
    try:
        safe_db_name = validate_identifier(config.database, SAFE_DATABASE_NAME_PATTERN)

        system_pool = await asyncpg.create_pool(
            host=config.host,
            port=config.port,
            database='postgres',
            user=config.user,
            password=config.password,
            min_size=1,
            max_size=2,
            timeout=30.0,
            command_timeout=60.0
        )

        async with system_pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_catalog.pg_database WHERE datname = $1)",
                safe_db_name
            )

            if not exists:
                await conn.execute(f'CREATE DATABASE "{safe_db_name}"')
            else:
                pass  # База уже существует

        await system_pool.close()
        return Result.success(None)
    except Exception as e:
        return Result.failure(e)


async def check_and_create_extension(
        pool: asyncpg.Pool,
        extension_name: str
) -> Result[bool]:
    """Проверяет наличие расширения и создаёт его при отсутствии."""
    try:
        safe_extension_name = validate_identifier(extension_name)

        async with pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = $1)",
                safe_extension_name
            )

            if not exists:
                await conn.execute(f"CREATE EXTENSION IF NOT EXISTS {safe_extension_name}")
                return Result.success(True)

            return Result.success(False)
    except Exception as e:
        return Result.failure(e)


async def create_tables_if_not_exists(pool: asyncpg.Pool, schema_config: DBSchemaConfig) -> Result[None]:
    """Создаёт все необходимые таблицы для RAG сервиса."""
    try:
        async with pool.acquire() as conn:
            # Таблица для отслеживания файлов
            await conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                hash VARCHAR(64) PRIMARY KEY,
                file_path TEXT NOT NULL,
                last_modified TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                file_size BIGINT NOT NULL CHECK (file_size >= 0),
                file_type VARCHAR(50) NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'active',

                CONSTRAINT valid_hash CHECK (hash ~ '^[a-fA-F0-9]{64}$'),
                CONSTRAINT valid_file_path CHECK (file_path <> ''),
                CONSTRAINT valid_file_type CHECK (file_type ~ '^[a-zA-Z0-9/._-]+$'),
                CONSTRAINT valid_status CHECK (status IN ('active', 'archived', 'deleted'))
            )
            """)

            # Основная таблица библиотеки
            await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS library (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content TEXT NOT NULL CHECK (content <> ''),
                metadata JSONB DEFAULT '{{}}'::jsonb,
                embedding VECTOR({schema_config.embedding_dimension}) NOT NULL,
                file_hash VARCHAR(64) NOT NULL REFERENCES files(hash) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """)

            # Таблица-очередь для векторизации
            await conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_queue (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content TEXT NOT NULL CHECK (content <> ''),
                metadata JSONB DEFAULT '{}'::jsonb,
                file_hash VARCHAR(64) NOT NULL REFERENCES files(hash) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
                status VARCHAR(20) NOT NULL DEFAULT 'pending',
                retry_count INTEGER DEFAULT 0 CHECK (retry_count >= 0),
                last_error TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                process_after TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

                CONSTRAINT valid_status CHECK (status IN ('pending', 'processing', 'completed', 'failed'))
            )
            """)

            return Result.success(None)
    except Exception as e:
        return Result.failure(e)


async def create_indexes_if_not_exists(
        pool: asyncpg.Pool,
        schema_config: DBSchemaConfig
) -> Result[None]:
    """Создаёт индексы для оптимизации векторного поиска."""
    try:
        async with pool.acquire() as conn:
            # Стандартные индексы
            await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_library_file_hash
            ON library (file_hash)
            """)

            await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunk_queue_file_hash
            ON chunk_queue (file_hash)
            """)

            await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunk_queue_processing
            ON chunk_queue (process_after, retry_count)
            WHERE status IN ('pending', 'failed')
            """)

            # Векторный индекс
            index_type = schema_config.vector_index_type.lower().strip()

            if index_type == 'hnsw':
                index_params = "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
            elif index_type == 'ivfflat':
                index_params = "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            else:
                index_params = "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"

            await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_library_embedding
            ON library {index_params}
            """)

            return Result.success(None)
    except Exception as e:
        return Result.failure(e)


# ======================================
# ГЛАВНАЯ ФУНКЦИЯ ИНИЦИАЛИЗАЦИИ
# ======================================

async def initialize_database() -> Result[None]:
    """Основной обработчик инициализации базы данных.

    Workflow:
    1. Загрузка конфигурации из .env файла
    2. Проверка и создание базы данных
    3. Подключение к целевой БД
    4. Создание необходимых расширений
    5. Создание таблиц с constraints
    6. Создание оптимизированных индексов

    Returns:
        Result[None]: Успешный результат или ошибка с деталями
    """
    try:
        env_path = Path.cwd() / '.env'
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
        else:
            load_dotenv(override=True)

        db_config = DBConfig()
        schema_config = DBSchemaConfig()

        # 1. Проверка и создание БД
        db_result = await ensure_database_exists(db_config)
        if not db_result.is_success():
            return db_result

        # 2. Подключение к целевой БД
        connection_result = await create_db_connection(db_config)
        if not connection_result.is_success():
            return connection_result

        pool = connection_result.unwrap()

        try:
            # 3. Проверка и создание расширений
            required_extensions = ['vector', 'pgcrypto']
            for ext in required_extensions:
                result = await check_and_create_extension(pool, ext)
                if not result.is_success():
                    return result

            # 4. Создание таблиц
            tables_result = await create_tables_if_not_exists(pool, schema_config)
            if not tables_result.is_success():
                return tables_result

            # 5. Создание индексов
            indexes_result = await create_indexes_if_not_exists(pool, schema_config)
            if not indexes_result.is_success():
                return indexes_result

            return Result.success(None)

        finally:
            await pool.close()

    except Exception as e:
        return Result.failure(e)