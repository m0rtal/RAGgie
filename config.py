"""Конфигурационные модели для инициализации базы данных.

Key guarantees:
- Централизованное управление конфигурацией
- Валидация параметров через Pydantic
- Поддержка 12-factor app принципов
"""

from pydantic import BaseModel, Field, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class DBConfig(BaseSettings):
    """Конфигурация подключения к PostgreSQL."""
    model_config = SettingsConfigDict(env_file='.env', env_prefix='DB_', extra='ignore')

    host: str = Field(default='localhost', description="Хост базы данных")
    port: int = Field(default=5432, description="Порт базы данных")
    database: str = Field(default='rag_db', description="Имя базы данных")
    user: str = Field(..., description="Пользователь базы данных")
    password: str = Field(..., description="Пароль пользователя")
    min_connections: int = Field(default=1, description="Минимальное количество соединений")
    max_connections: int = Field(default=5, description="Максимальное количество соединений")


class DBSchemaConfig(BaseModel):
    """Конфигурация схемы базы данных."""
    embedding_dimension: PositiveInt = Field(
        default=1536,
        description="Размерность вектора эмбеддинга"
    )
    full_text_languages: List[str] = Field(
        default=['russian', 'english'],
        description="Поддерживаемые языки для полнотекстового поиска"
    )
    vector_index_type: str = Field(
        default='hnsw',
        description="Тип векторного индекса (hnsw или ivfflat)"
    )