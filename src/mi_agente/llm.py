"""Cliente LLM unificado con soporte multi-proveedor."""

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

from mi_agente.config import Settings, get_settings


def _parse_tool_args(arguments: object) -> dict:
    """Convierte los argumentos de tool call a dict (Anthropic requiere objeto)."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


@dataclass
class Message:
    """Mensaje de conversación."""

    role: str  # system, user, assistant, tool
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # para tool messages

    def to_openai(self) -> dict:
        if self.role == "system":
            return {"role": "system", "content": self.content or ""}
        msg: dict[str, Any] = {"role": self.role}
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
        content: str | list[dict[str, Any]] = self.content or ""
        if self.tool_calls:
            # Anthropic usa tool_use en content
            text_block: dict[str, Any] | None = (
                {"type": "text", "text": content} if isinstance(content, str) and content else None
            )
            tool_blocks: list[dict[str, Any]] = [
                {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": _parse_tool_args(tc["function"].get("arguments")),
                }
                for tc in self.tool_calls
            ]
            content_list: list[dict[str, Any]] = []
            if text_block is not None:
                content_list.append(text_block)
            content_list.extend(tool_blocks)
            content = content_list
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
        **kwargs: Any,
    ) -> LLMResponse | Iterator[LLMResponse]:
        """Completa la conversación."""

    @abstractmethod
    async def acomplete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse | AsyncIterator[LLMResponse]:
        """Completa la conversación (async)."""
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, messages: list[Message]) -> int:
        """Cuenta tokens aproximados."""
        raise NotImplementedError


def create_llm_client(settings: Settings | None = None) -> LLMClient:
    """Factory para crear cliente LLM según configuración."""
    settings = settings or get_settings()

    if settings.llm_provider == "openai":
        from mi_agente.llm_openai import OpenAIClient

        return OpenAIClient(settings)
    if settings.llm_provider == "anthropic":
        from mi_agente.llm_anthropic import AnthropicClient

        return AnthropicClient(settings)
    if settings.llm_provider == "ollama":
        from mi_agente.llm_ollama import OllamaClient

        return OllamaClient(settings)
    raise ValueError(f"Proveedor desconocido: {settings.llm_provider}")
