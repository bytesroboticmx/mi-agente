"""Herramienta para ejecutar comandos shell."""

import subprocess
from pathlib import Path

from mi_agente.tools.base import Tool, ToolResult


class ShellCommand(Tool):
    """Ejecuta un comando en el shell."""

    name = "shell"
    description = "Ejecuta un comando en el shell del sistema"
    dangerous = True
    timeout = 60
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Comando a ejecutar"},
            "cwd": {"type": "string", "description": "Directorio de trabajo", "default": "."},
            "timeout": {"type": "integer", "description": "Timeout en segundos", "default": 30},
        },
        "required": ["command"],
    }

    def execute(self, command: str, cwd: str = ".", timeout: int = 30) -> ToolResult:  # type: ignore[override]
        try:
            work_dir = Path(cwd).expanduser().resolve()
            if not work_dir.exists():
                return ToolResult(
                    success=False, content="", error=f"Directorio no encontrado: {cwd}"
                )

            # Usar shell=True para soportar pipes, redirecciones, etc.
            result = subprocess.run(
                command,
                shell=True,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

            output = []
            if result.stdout:
                output.append(f"[stdout]\n{result.stdout}")
            if result.stderr:
                output.append(f"[stderr]\n{result.stderr}")
            output.append(f"[exit code: {result.returncode}]")

            success = result.returncode == 0
            return ToolResult(success=success, content="\n".join(output))
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False, content="", error=f"Timeout ({timeout}s) ejecutando: {command}"
            )
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))
