"""Tests para configuración."""

import pytest

from mi_agente.config import Settings, _load_env_file, _parse_value, get_settings, reload_settings


def test_settings_defaults():
    """Verifica valores por defecto."""
    s = Settings()
    assert s.llm_provider == "openai"
    assert s.model_name == "gpt-4o-mini"
    assert s.temperature == 0.7
    assert s.max_tokens == 4096
    assert s.memory_enabled is True
    assert s.confirm_dangerous_tools is True
    assert s.theme == "default"


def test_settings_validation():
    """Verifica validación de campos."""
    # Temperatura fuera de rango
    with pytest.raises(ValueError):
        Settings(temperature=3.0)

    with pytest.raises(ValueError):
        Settings(temperature=-0.5)

    # Max tokens
    with pytest.raises(ValueError):
        Settings(max_tokens=0)

    with pytest.raises(ValueError):
        Settings(max_tokens=200000)

    # Provider inválido
    with pytest.raises(ValueError):
        Settings(llm_provider="invalid")


def test_parse_value():
    """Verifica parsing de valores desde .env."""
    assert _parse_value("TEMPERATURE", "0.5") == 0.5
    assert _parse_value("MAX_TOKENS", "1000") == 1000
    assert _parse_value("TOOL_TIMEOUT", "60") == 60
    assert _parse_value("CONFIRM_DANGEROUS_TOOLS", "true") is True
    assert _parse_value("CONFIRM_DANGEROUS_TOOLS", "false") is False
    assert _parse_value("CONFIRM_DANGEROUS_TOOLS", "1") is True
    assert _parse_value("CONFIRM_DANGEROUS_TOOLS", "0") is False
    assert _parse_value("MEMORY_ENABLED", "yes") is True
    assert _parse_value("MEMORY_ENABLED", "no") is False
    assert _parse_value("OTHER", "string") == "string"


def test_load_env_file(tmp_path):
    """Verifica carga de archivo .env."""
    env_file = tmp_path / ".env"
    env_file.write_text("""
LLM_PROVIDER=anthropic
MODEL_NAME=claude-3
API_KEY=sk-test
TEMPERATURE=0.5
MAX_TOKENS=2000
""")

    data = _load_env_file(env_file)
    assert data["LLM_PROVIDER"] == "anthropic"
    assert data["MODEL_NAME"] == "claude-3"
    assert data["API_KEY"] == "sk-test"
    assert data["TEMPERATURE"] == "0.5"
    assert data["MAX_TOKENS"] == "2000"


def test_get_settings_from_env(tmp_path, monkeypatch):
    """Verifica get_settings carga desde .env."""
    env_file = tmp_path / ".env"
    env_file.write_text("""
LLM_PROVIDER=ollama
MODEL_NAME=llama3
TEMPERATURE=0.3
""")

    monkeypatch.chdir(tmp_path)
    reload_settings()
    s = get_settings()

    assert s.llm_provider == "ollama"
    assert s.model_name == "llama3"
    assert s.temperature == 0.3


def test_secret_str_api_key():
    """Verifica manejo de API key secreta."""
    s = Settings(api_key="sk-secret")
    assert s.get_api_key() == "sk-secret"

    s2 = Settings(api_key="")
    assert s2.get_api_key() == ""


def test_path_expansion():
    """Verifica expansión de paths."""
    s = Settings(working_dir="~/projects", memory_dir="~/.my-agent/mem")
    assert str(s.working_dir).startswith("/")  # expandido
    assert str(s.memory_dir).startswith("/")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
