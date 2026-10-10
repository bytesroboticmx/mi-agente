"""mi-agente: Agente CLI inteligente para terminal Linux."""

__version__ = "0.1.0"
__author__ = "Tu Nombre"
__email__ = "tu@email.com"

from mi_agente.agent import Agent
from mi_agente.config import Settings, get_settings
from mi_agente.llm import LLMClient, Message, ToolSpec, create_llm_client
from mi_agente.tools import Tool, ToolRegistry, ToolResult, create_default_registry

__all__ = [
    "Agent",
    "Settings",
    "get_settings",
    "LLMClient",
    "Message",
    "ToolSpec",
    "create_llm_client",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "create_default_registry",
]