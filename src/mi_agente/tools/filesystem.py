"""Herramientas de sistema de archivos."""

from pathlib import Path

from mi_agente.tools.base import Tool, ToolResult


class ReadFile(Tool):
    """Lee el contenido de un archivo."""

    name = "read_file"
    description = "Lee el contenido de un archivo de texto"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Ruta del archivo a leer"},
            "offset": {"type": "integer", "description": "Línea inicial (0-indexed)", "default": 0},
            "limit": {
                "type": "integer",
                "description": "Número de líneas a leer (0 = todo)",
                "default": 0,
            },
        },
        "required": ["path"],
    }

    def execute(self, path: str, offset: int = 0, limit: int = 0) -> ToolResult:  # type: ignore[override]
        try:
            file_path = Path(path).expanduser().resolve()
            if not file_path.exists():
                return ToolResult(success=False, content="", error=f"Archivo no encontrado: {path}")
            if not file_path.is_file():
                return ToolResult(success=False, content="", error=f"No es un archivo: {path}")

            content = file_path.read_text(encoding="utf-8")
            lines = content.splitlines()

            if offset > 0:
                lines = lines[offset:]
            if limit > 0:
                lines = lines[:limit]

            result = "\n".join(lines)
            if offset > 0 or limit > 0:
                total = len(content.splitlines())
                result += f"\n\n[Mostrando líneas {offset}-{offset + len(lines)} de {total}]"

            return ToolResult(success=True, content=result)
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                content="",
                error=f"Archivo binario o codificación no soportada: {path}",
            )
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))


class WriteFile(Tool):
    """Escribe contenido a un archivo."""

    name = "write_file"
    description = "Escribe contenido a un archivo (crea directorios si no existen)"
    dangerous = True
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Ruta del archivo a escribir"},
            "content": {"type": "string", "description": "Contenido a escribir"},
            "append": {
                "type": "boolean",
                "description": "Añadir al final en lugar de sobrescribir",
                "default": False,
            },
        },
        "required": ["path", "content"],
    }

    def execute(self, path: str, content: str, append: bool = False) -> ToolResult:  # type: ignore[override]
        try:
            file_path = Path(path).expanduser().resolve()
            file_path.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append else "w"
            with file_path.open(mode, encoding="utf-8") as f:
                f.write(content)

            action = "Añadido" if append else "Escrito"
            return ToolResult(success=True, content=f"{action} {len(content)} caracteres a {path}")
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))


class ListDir(Tool):
    """Lista contenido de un directorio."""

    name = "list_dir"
    description = "Lista archivos y directorios en una ruta"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Ruta del directorio", "default": "."},
            "recursive": {
                "type": "boolean",
                "description": "Listar recursivamente",
                "default": False,
            },
            "show_hidden": {
                "type": "boolean",
                "description": "Mostrar archivos ocultos",
                "default": False,
            },
        },
        "required": [],
    }

    def execute(  # type: ignore[override]
        self, path: str = ".", recursive: bool = False, show_hidden: bool = False
    ) -> ToolResult:
        try:
            dir_path = Path(path).expanduser().resolve()
            if not dir_path.exists():
                return ToolResult(
                    success=False, content="", error=f"Directorio no encontrado: {path}"
                )
            if not dir_path.is_dir():
                return ToolResult(success=False, content="", error=f"No es un directorio: {path}")

            if recursive:
                items = list(dir_path.rglob("*"))
            else:
                items = list(dir_path.iterdir())

            if not show_hidden:
                items = [i for i in items if not i.name.startswith(".")]

            lines = []
            for item in sorted(items):
                rel = item.relative_to(dir_path)
                prefix = "📁 " if item.is_dir() else "📄 "
                size = f" ({item.stat().st_size} bytes)" if item.is_file() else ""
                lines.append(f"{prefix}{rel}{size}")

            return ToolResult(success=True, content="\n".join(lines) if lines else "(vacío)")
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))


class Glob(Tool):
    """Busca archivos por patrón glob."""

    name = "glob"
    description = "Busca archivos que coincidan con un patrón glob"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Patrón glob (ej: **/*.py)"},
            "root": {"type": "string", "description": "Directorio raíz", "default": "."},
        },
        "required": ["pattern"],
    }

    def execute(self, pattern: str, root: str = ".") -> ToolResult:  # type: ignore[override]
        try:
            root_path = Path(root).expanduser().resolve()
            matches = list(root_path.rglob(pattern))
            rel_paths = [str(m.relative_to(root_path)) for m in matches]
            return ToolResult(
                success=True, content="\n".join(rel_paths) if rel_paths else "(sin coincidencias)"
            )
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))


class Grep(Tool):
    """Busca texto en archivos."""

    name = "grep"
    description = "Busca un patrón de texto en archivos"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Patrón de búsqueda (regex)"},
            "path": {
                "type": "string",
                "description": "Archivo o directorio a buscar",
                "default": ".",
            },
            "include": {"type": "string", "description": "Patrón de archivos a incluir (glob)"},
        },
        "required": ["pattern"],
    }

    def execute(self, pattern: str, path: str = ".", include: str | None = None) -> ToolResult:  # type: ignore[override]
        try:
            import re

            search_path = Path(path).expanduser().resolve()

            if search_path.is_file():
                files = [search_path]
            elif include:
                files = list(search_path.rglob(include))
            else:
                files = [f for f in search_path.rglob("*") if f.is_file()]

            regex = re.compile(pattern)
            results = []

            for file in files:
                try:
                    content = file.read_text(encoding="utf-8")
                    for i, line in enumerate(content.splitlines(), 1):
                        if regex.search(line):
                            rel = file.relative_to(search_path) if search_path.is_dir() else file
                            results.append(f"{rel}:{i}: {line.strip()}")
                except UnicodeDecodeError:
                    continue
                except Exception:
                    continue

            return ToolResult(
                success=True, content="\n".join(results) if results else "(sin coincidencias)"
            )
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))
