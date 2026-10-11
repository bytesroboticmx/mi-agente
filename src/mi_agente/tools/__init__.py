"""Inicialización de herramientas."""

from mi_agente.tools.base import Tool, ToolRegistry, ToolResult
from mi_agente.tools.filesystem import Glob, Grep, ListDir, ReadFile, WriteFile
from mi_agente.tools.shell import ShellCommand

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "ReadFile",
    "WriteFile",
    "ListDir",
    "Glob",
    "Grep",
    "ShellCommand",
]


def create_default_registry() -> ToolRegistry:
    """Crea registro con herramientas por defecto."""
    registry = ToolRegistry()
    registry.register(ReadFile())
    registry.register(WriteFile())
    registry.register(ListDir())
    registry.register(Glob())
    registry.register(Grep())
    registry.register(ShellCommand())
    return registry
