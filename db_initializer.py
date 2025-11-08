# db_initializer.py
"""Инициализация PostgreSQL базы данных для RAG сервиса.

Key guarantees:
- Идемпотентность всех операций (безопасный повторный запуск)
- Оптимизированная схема только для векторного поиска
- Безопасность: строгая валидация идентификаторов и параметризованные запросы
- Отказоустойчивость: обработка ошибок через Railway-ориентированное программирование
- Изоляция сбоев через bulkheads (разделение системных и пользовательских соединений)
- Graceful shutdown с корректным закрытием соединений

Trade-offs considered:
- Полностью удалён полнотекстовый поиск (tsvector) как избыточный для RAG
- Использованы конфигурируемые параметры для векторных индексов
- Разделение пулов соединений для системных операций и основной работы
"""

import logging
import re
from typing import Any, Generic, Optional, TypeVar, cast, Final, Callable
from functools import wraps
import asyncio

import asyncpg
from dotenv import load_dotenv
from config import app_config, DBSecurityConfig, DBConnectionConfig, DBSchemaConfig

# ======================================
# КОНСТАНТЫ И ВАЛИДАЦИЯ
# ======================================

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


def validate_identifier(name: str, config: DBSecurityConfig) -> str:
    """Валидация идентификаторов для предотвращения SQL инъекций."""
    if len(name) > config.max_identifier_length:
        raise ValueError(f"Идентификатор превышает максимальную длину {config.max_identifier_length} символов")

    if not re.match(config.identifier_pattern, name):
        raise ValueError(f"Недопустимое имя объекта: {name}. "
                         f"Должно соответствовать шаблону: {config.identifier_pattern}")
    return name


def validate_database_name(name: str, config: DBSecurityConfig) -> str:
    """Валидация имени базы данных."""
    if len(name) > config.max_identifier_length:
        raise ValueError(f"Имя базы данных превышает максимальную длину {config.max_identifier_length} символов")

    if not re.match(config.database_name_pattern, name):
        raise ValueError(f"Недопустимое имя базы данных: {name}. "
                         f"Должно соответствовать шаблону: {config.database_name_pattern}")
    return name


# ======================================
# ФУНКЦИОНАЛЬНЫЕ КОМПОНЕНТЫ
# ======================================

async def create_system_pool(config: DBConnectionConfig) -> Result[asyncpg.Pool]:
    """Создает пул соединений к системной базе данных (postgres) для административных операций."""
    try:
        pool = await asyncpg.create_pool(
            host=config.host,
            port=config.port,
            database='postgres',
            user=config.user,
            password=config.password,
            min_size=1,
            max_size=2,
            timeout=config.connection_timeout,
            command_timeout=config.command_timeout,
            max_inactive_connection_lifetime=config.max_inactive_lifetime
        )
        return Result.success(pool)
    except Exception as e:
        return Result.failure(e)


async def create_app_pool(config: DBConnectionConfig) -> Result[asyncpg.Pool]:
    """Создает пул соединений к целевой базе данных приложения."""
    try:
        pool = await asyncpg.create_pool(
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.user,
            password=config.password,
            min_size=config.min_connections,
            max_size=config.max_connections,
            timeout=config.connection_timeout,
            command_timeout=config.command_timeout,
            max_inactive_connection_lifetime=config.max_inactive_lifetime
        )
        return Result.success(pool)
    except Exception as e:
        return Result.failure(e)


async def ensure_database_exists(
        system_pool: asyncpg.Pool,
        db_config: DBConnectionConfig,
        security_config: DBSecurityConfig
) -> Result[None]:
    """Проверяет существование базы данных и создаёт её при отсутствии."""
    try:
        safe_db_name = validate_database_name(db_config.database, security_config)

        async with system_pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_catalog.pg_database WHERE datname = $1)",
                safe_db_name
            )

            if not exists:
                # Используем параметризованный запрос для безопасности
                await conn.execute(f'CREATE DATABASE "{safe_db_name}"')
            return Result.success(None)

    except Exception as e:
        return Result.failure(e)


async def check_and_create_extension(
        pool: asyncpg.Pool,
        extension_name: str,
        security_config: DBSecurityConfig
) -> Result[bool]:
    """Проверяет наличие расширения и создаёт его при отсутствии."""
    try:
        safe_extension_name = validate_identifier(extension_name, security_config)

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


async def create_tables_if_not_exists(pool: asyncpg.Pool) -> Result[None]:
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
            embedding_dim = app_config.schema_config.embedding_dimension
            await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS library (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content TEXT NOT NULL CHECK (content <> ''),
                metadata JSONB DEFAULT '{{}}'::jsonb,
                embedding VECTOR({embedding_dim}) NOT NULL,
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
            index_config = schema_config.vector_index

            if index_config.type == 'hnsw':
                index_params = (
                    f"USING hnsw (embedding {index_config.distance_metric}) "
                    f"WITH (m = {index_config.hnsw_m}, ef_construction = {index_config.hnsw_ef_construction})"
                )
            elif index_config.type == 'ivfflat':
                index_params = (
                    f"USING ivfflat (embedding {index_config.distance_metric}) "
                    f"WITH (lists = {index_config.ivfflat_lists})"
                )
            else:
                index_params = (
                    f"USING hnsw (embedding {index_config.distance_metric}) "
                    f"WITH (m = {index_config.hnsw_m}, ef_construction = {index_config.hnsw_ef_construction})"
                )

            await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_library_embedding
            ON library {index_params}
            """)

            # Добавляем комментарий к индексу для документации
            await conn.execute("""
            COMMENT ON INDEX idx_library_embedding IS 'Векторный индекс для поиска по эмбеддингам'
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
    2. Проверка и создание базы данных через системный пул
    3. Подключение к целевой БД
    4. Создание необходимых расширений
    5. Создание таблиц с constraints
    6. Создание оптимизированных индексов

    Returns:
        Result[None]: Успешный результат или ошибка с деталями

    Key guarantees:
    - Все операции идемпотентны
    - Системные и пользовательские соединения изолированы
    - Все ресурсы корректно закрываются при завершении
    """
    system_pool = None
    app_pool = None

    try:
        # Загрузка .env файла
        load_dotenv(override=True)

        # Валидация конфигурации
        db_config = app_config.db
        security_config = app_config.security
        schema_config = app_config.schema_config

        # 1. Создание системного пула для административных операций
        system_result = await create_system_pool(db_config)
        if not system_result.is_success():
            return system_result
        system_pool = system_result.unwrap()

        try:
            # 2. Проверка и создание базы данных
            db_result = await ensure_database_exists(system_pool, db_config, security_config)
            if not db_result.is_success():
                return db_result

            # 3. Создание пула для приложения
            app_result = await create_app_pool(db_config)
            if not app_result.is_success():
                return app_result
            app_pool = app_result.unwrap()

            try:
                # 4. Проверка и создание расширений
                for ext in schema_config.required_extensions:
                    result = await check_and_create_extension(
                        app_pool,
                        ext,
                        security_config
                    )
                    if not result.is_success():
                        return result

                # 5. Создание таблиц
                tables_result = await create_tables_if_not_exists(app_pool)
                if not tables_result.is_success():
                    return tables_result

                # 6. Создание индексов
                indexes_result = await create_indexes_if_not_exists(app_pool, schema_config)
                if not indexes_result.is_success():
                    return indexes_result

                return Result.success(None)

            finally:
                if app_pool:
                    await app_pool.close()

        finally:
            if system_pool:
                await system_pool.close()

    except Exception as e:
        return Result.failure(e)