"""Base para herramientas del agente."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Resultado de ejecutar una herramienta."""

    success: bool
    content: str
    error: str | None = None
    metadata: dict | None = None

    def __str__(self) -> str:
        return self.content


class Tool(ABC):
    """Clase base para herramientas."""

    name: str
    description: str
    parameters: dict  # JSON Schema
    dangerous: bool = False  # Requiere confirmación
    timeout: int = 30

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Ejecuta la herramienta."""

    def to_openai_format(self) -> dict:
        """Formato para OpenAI function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_format(self) -> dict:
        """Formato para Anthropic tool use."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def validate_args(self, args: dict) -> tuple[bool, str | None]:
        """Valida argumentos contra el schema (básico)."""
        # TODO: Validación completa con jsonschema
        required = self.parameters.get("required", [])
        for field in required:
            if field not in args:
                return False, f"Falta argumento requerido: {field}"
        return True, None


class ToolRegistry:
    """Registro de herramientas disponibles."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def get_openai_tools(self) -> list[dict]:  # type: ignore[valid-type]
        return [t.to_openai_format() for t in self._tools.values()]

    def get_anthropic_tools(self) -> list[dict]:  # type: ignore[valid-type]
        return [t.to_anthropic_format() for t in self._tools.values()]

    def execute(self, name: str, args: dict, confirm: bool = True) -> ToolResult:
        """Ejecuta una herramienta por nombre."""
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, content="", error=f"Herramienta no encontrada: {name}")

        valid, error = tool.validate_args(args)
        if not valid:
            return ToolResult(success=False, content="", error=error)

        # Confirmación para tools peligrosas
        if confirm and tool.dangerous:
            from mi_agente.output import confirm as confirm_prompt

            if not confirm_prompt(f"Ejecutar {tool.name} con args: {args}?"):
                return ToolResult(success=False, content="", error="Cancelado por usuario")

        try:
            return tool.execute(**args)
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Error ejecutando {name}: {e}")
