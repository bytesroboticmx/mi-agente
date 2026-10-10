"""Cliente LLM unificado con soporte multi-proveedor."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator

from mi_agente.config import Settings, get_settings


@dataclass
class Message:
    """Mensaje de conversación."""
    role: str  # system, user, assistant, tool
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # para tool messages

    def to_openai(self) -> dict:
        msg = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg

    def to_anthropic(self) -> dict:
        if self.role == "system":
            return {"role": "system", "content": self.content or ""}
        if self.role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": self.tool_call_id,
                        "content": self.content or "",
                    }
                ],
            }
        content = self.content or ""
        if self.tool_calls:
            # Anthropic usa tool_use en content
            content = [
                {"type": "text", "text": content} if content else None,
                *[
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": tc["function"]["arguments"],
                    }
                    for tc in self.tool_calls
                ],
            ]
            content = [c for c in content if c is not None]
        return {"role": "assistant" if self.role == "assistant" else "user", "content": content}


@dataclass
class ToolSpec:
    """Especificación de herramienta para function calling."""
    name: str
    description: str
    parameters: dict  # JSON Schema

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class LLMResponse:
    """Respuesta del LLM."""
    content: str | None = None
    tool_calls: list[dict] | None = None
    usage: dict | None = None
    model: str | None = None
    finish_reason: str | None = None


class LLMClient(ABC):
    """Interfaz base para clientes LLM."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse | Iterator[LLMResponse]:
        """Completa la conversación."""
        pass

    @abstractmethod
    async def acomplete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse | AsyncIterator[LLMResponse]:
        """Completa la conversación (async)."""
        pass

    @abstractmethod
    def count_tokens(self, messages: list[Message]) -> int:
        """Cuenta tokens aproximados."""
        pass


def create_llm_client(settings: Settings | None = None) -> LLMClient:
    """Factory para crear cliente LLM según configuración."""
    settings = settings or get_settings()

    if settings.llm_provider == "openai":
        from mi_agente.llm_openai import OpenAIClient
        return OpenAIClient(settings)
    elif settings.llm_provider == "anthropic":
        from mi_agente.llm_anthropic import AnthropicClient
        return AnthropicClient(settings)
    elif settings.llm_provider == "ollama":
        from mi_agente.llm_ollama import OllamaClient
        return OllamaClient(settings)
    else:
        raise ValueError(f"Proveedor desconocido: {settings.llm_provider}")