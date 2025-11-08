# config.py
"""Конфигурационные модели для инициализации базы данных.

Key guarantees:
- Централизованное управление конфигурацией
- Валидация параметров через Pydantic V2
- Поддержка 12-factor app принципов
- Безопасные значения по умолчанию для всех параметров
"""

from pydantic import BaseModel, Field, PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Literal, Optional
from pathlib import Path


class DBSecurityConfig(BaseModel):
    """Параметры безопасности базы данных."""
    identifier_pattern: str = Field(
        default=r'^[a-zA-Z0-9_]+$',
        description="Регулярное выражение для валидации идентификаторов"
    )
    database_name_pattern: str = Field(
        default=r'^[a-zA-Z0-9_-]+$',
        description="Регулярное выражение для валидации имен баз данных"
    )
    max_identifier_length: int = Field(
        default=63,
        description="Максимальная длина идентификаторов PostgreSQL"
    )

    @field_validator('identifier_pattern', 'database_name_pattern')
    @classmethod
    def validate_patterns(cls, v: str) -> str:
        """Проверяет корректность регулярных выражений."""
        import re
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"Некорректное регулярное выражение: {str(e)}")
        return v


class DBConnectionConfig(BaseSettings):
    """Конфигурация подключения к PostgreSQL."""
    model_config = SettingsConfigDict(env_file='.env', env_prefix='DB_', extra='ignore')

    host: str = Field(default='localhost', description="Хост базы данных")
    port: int = Field(default=5432, description="Порт базы данных", ge=1, le=65535)
    database: str = Field(default='rag_db', description="Имя базы данных")
    user: str = Field(..., description="Пользователь базы данных")
    password: str = Field(..., description="Пароль пользователя", min_length=1)
    min_connections: int = Field(default=1, description="Минимальное количество соединений", ge=1)
    max_connections: int = Field(default=5, description="Максимальное количество соединений", ge=1)
    connection_timeout: float = Field(default=30.0, description="Таймаут подключения в секундах", gt=0)
    command_timeout: float = Field(default=60.0, description="Таймаут команды в секундах", gt=0)
    max_inactive_lifetime: float = Field(
        default=300.0,
        description="Максимальное время неактивности соединения в секундах",
        gt=0
    )

    @field_validator('database')
    @classmethod
    def validate_database_name(cls, v: str) -> str:
        """Валидация имени базы данных."""
        if len(v) > 63:
            raise ValueError("Имя базы данных не должно превышать 63 символа")
        if not v.replace('_', '').isalnum():
            raise ValueError("Имя базы данных должно содержать только буквы, цифры и подчеркивания")
        return v


class VectorIndexConfig(BaseModel):
    """Конфигурация векторных индексов."""
    type: Literal['hnsw', 'ivfflat'] = Field(
        default='hnsw',
        description="Тип векторного индекса"
    )
    hnsw_m: int = Field(
        default=16,
        description="Параметр M для HNSW индекса",
        ge=2,
        le=100
    )
    hnsw_ef_construction: int = Field(
        default=64,
        description="Параметр ef_construction для HNSW индекса",
        ge=4,
        le=1000
    )
    ivfflat_lists: int = Field(
        default=100,
        description="Количество списков для IVFFlat индекса",
        ge=10,
        le=1000
    )
    distance_metric: Literal['vector_cosine_ops', 'vector_l2_ops', 'vector_ip_ops'] = Field(
        default='vector_cosine_ops',
        description="Метрика расстояния для векторного поиска"
    )


class DBSchemaConfig(BaseModel):
    """Конфигурация схемы базы данных."""
    embedding_dimension: PositiveInt = Field(
        default=1536,
        description="Размерность вектора эмбеддинга",
        ge=1,
        le=4096
    )
    vector_index: VectorIndexConfig = Field(default_factory=VectorIndexConfig)
    required_extensions: List[str] = Field(
        default=['vector', 'pgcrypto'],
        description="Требуемые расширения PostgreSQL"
    )


class LoggingConfig(BaseSettings):
    """Конфигурация логирования."""
    model_config = SettingsConfigDict(env_file='.env', env_prefix='LOG_', extra='ignore')

    level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(
        default='INFO',
        description="Уровень логирования"
    )
    file_path: Path = Field(
        default=Path('db_init.log'),
        description="Путь к файлу логов"
    )
    console_enabled: bool = Field(default=True, description="Включить вывод в консоль")
    third_party_levels: dict = Field(
        default_factory=lambda: {
            'asyncpg': 'WARNING',
            'dotenv': 'INFO'
        },
        description="Уровни логирования для сторонних библиотек"
    )


class AppConfig(BaseSettings):
    """Общая конфигурация приложения."""
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    security: DBSecurityConfig = Field(default_factory=DBSecurityConfig)
    db: DBConnectionConfig = Field(default_factory=DBConnectionConfig)
    schema_config: DBSchemaConfig = Field(default_factory=DBSchemaConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# Экземпляр конфигурации для использования в приложении
app_config = AppConfig()