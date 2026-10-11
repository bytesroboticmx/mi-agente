"""Configuración del agente usando Pydantic + python-dotenv (sin pydantic-settings)."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class Settings(BaseModel):
    """Configuración global del agente."""

    model_config = ConfigDict(extra="ignore")

    # LLM Provider
    llm_provider: Literal["openai", "anthropic", "ollama"] = "openai"

    # Modelo
    model_name: str = "gpt-4o-mini"

    # API Key (secreto)
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))

    # Base URL opcional (para Ollama/local)
    base_url: str | None = None

    # Parámetros de generación
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0, le=128000)

    # System prompt
    system_prompt: str = Field(
        default=(
            "Eres un asistente útil que puede usar herramientas para "
            "leer/escribir archivos, ejecutar comandos y buscar información."
        )
    )

    # Directorio de trabajo
    working_dir: Path = Field(default_factory=Path.cwd)

    # Configuración de tools
    tool_timeout: int = Field(default=30, gt=0)
    confirm_dangerous_tools: bool = True

    # Memoria
    memory_enabled: bool = True
    memory_dir: Path = Field(default_factory=lambda: Path.home() / ".mi-agente" / "memory")

    # UI / Tema
    theme: Literal["default", "dark", "light"] = "default"

    @field_validator("working_dir", "memory_dir", mode="before")
    @classmethod
    def expand_path(cls, v: object) -> object:
        if isinstance(v, str):
            return Path(v).expanduser()
        return v

    def get_api_key(self) -> str:
        """Obtiene la API key como string plano."""
        return self.api_key.get_secret_value() if self.api_key else ""


_ENV_VARS = [
    "LLM_PROVIDER",
    "MODEL_NAME",
    "API_KEY",
    "BASE_URL",
    "TEMPERATURE",
    "MAX_TOKENS",
    "SYSTEM_PROMPT",
    "WORKING_DIR",
    "TOOL_TIMEOUT",
    "CONFIRM_DANGEROUS_TOOLS",
    "MEMORY_ENABLED",
    "MEMORY_DIR",
    "THEME",
]


def _clean_env_value(value: str) -> str:
    """Quita comillas y comentarios en línea de un valor .env."""
    value = value.strip()
    if not value:
        return value
    quoted = value[0] in ("'", '"') and value[-1] == value[0]
    if quoted:
        return value[1:-1]
    # Comentario en línea solo cuando el valor no está citado
    if "#" in value:
        value = value.split("#", 1)[0].rstrip()
    return value


def _load_env_file(env_path: Path = Path(".env")) -> dict:
    """Carga variables desde archivo .env manualmente."""
    result: dict[str, str] = {}
    if env_path.exists():
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            result[key.strip()] = _clean_env_value(value)
    # También incluir variables de entorno del sistema
    for key in _ENV_VARS:
        if key in os.environ:
            result[key] = os.environ[key]
    return result


def _parse_value(key: str, value: str) -> bool | int | float | str:
    """Convierte string a tipo apropiado."""
    # Boolean
    if key in ("CONFIRM_DANGEROUS_TOOLS", "MEMORY_ENABLED"):
        return value.lower() in ("true", "1", "yes", "on")
    # Integer
    if key in ("MAX_TOKENS", "TOOL_TIMEOUT"):
        return int(value)
    # Float
    if key == "TEMPERATURE":
        return float(value)
    return value


@lru_cache
def get_settings() -> Settings:
    """Obtiene la instancia de settings (cached)."""
    env_data = _load_env_file()
    # Mapear env vars a campos del modelo
    field_map = {
        "LLM_PROVIDER": "llm_provider",
        "MODEL_NAME": "model_name",
        "API_KEY": "api_key",
        "BASE_URL": "base_url",
        "TEMPERATURE": "temperature",
        "MAX_TOKENS": "max_tokens",
        "SYSTEM_PROMPT": "system_prompt",
        "WORKING_DIR": "working_dir",
        "TOOL_TIMEOUT": "tool_timeout",
        "CONFIRM_DANGEROUS_TOOLS": "confirm_dangerous_tools",
        "MEMORY_ENABLED": "memory_enabled",
        "MEMORY_DIR": "memory_dir",
        "THEME": "theme",
    }
    parsed: dict[str, object] = {}
    for key, value in env_data.items():
        field_name = field_map.get(key, key.lower())
        parsed[field_name] = _parse_value(key, value)
    return Settings(**parsed)  # type: ignore[arg-type]


def reload_settings() -> Settings:
    """Fuerza recarga de settings (limpia cache)."""
    get_settings.cache_clear()
    return get_settings()
