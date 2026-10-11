"""Tests para herramientas."""

import pytest

from mi_agente.tools import ToolRegistry, create_default_registry
from mi_agente.tools.filesystem import Glob, Grep, ListDir, ReadFile, WriteFile
from mi_agente.tools.shell import ShellCommand


def test_tool_registry():
    """Verifica registro de herramientas."""
    registry = ToolRegistry()
    tool = ReadFile()
    registry.register(tool)

    assert registry.get("read_file") == tool
    assert registry.get("nonexistent") is None
    assert len(registry.list()) == 1

    openai_tools = registry.get_openai_tools()
    assert len(openai_tools) == 1
    assert openai_tools[0]["function"]["name"] == "read_file"

    anthropic_tools = registry.get_anthropic_tools()
    assert len(anthropic_tools) == 1
    assert anthropic_tools[0]["name"] == "read_file"


def test_read_file(tmp_path):
    """Verifica lectura de archivos."""
    tool = ReadFile()
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hola mundo\nLínea 2\nLínea 3")

    result = tool.execute(path=str(test_file))
    assert result.success
    assert "Hola mundo" in result.content
    assert "Línea 2" in result.content

    # Test offset y limit
    result = tool.execute(path=str(test_file), offset=1, limit=1)
    assert result.success
    assert "Línea 2" in result.content
    assert "Hola mundo" not in result.content


def test_read_file_not_found():
    """Verifica error archivo no encontrado."""
    tool = ReadFile()
    result = tool.execute(path="/nonexistent/file.txt")
    assert not result.success
    assert "no encontrado" in result.error.lower()


def test_write_file(tmp_path):
    """Verifica escritura de archivos."""
    tool = WriteFile()
    test_file = tmp_path / "output.txt"

    result = tool.execute(path=str(test_file), content="Nuevo contenido")
    assert result.success
    assert test_file.read_text() == "Nuevo contenido"

    # Test append
    result = tool.execute(path=str(test_file), content="\nAñadido", append=True)
    assert result.success
    assert test_file.read_text() == "Nuevo contenido\nAñadido"


def test_write_file_creates_dirs(tmp_path):
    """Verifica que crea directorios padre."""
    tool = WriteFile()
    test_file = tmp_path / "subdir" / "nested" / "file.txt"

    result = tool.execute(path=str(test_file), content="test")
    assert result.success
    assert test_file.read_text() == "test"


def test_list_dir(tmp_path):
    """Verifica listado de directorio."""
    tool = ListDir()
    (tmp_path / "file1.txt").write_text("a")
    (tmp_path / "file2.py").write_text("b")
    (tmp_path / "subdir").mkdir()

    result = tool.execute(path=str(tmp_path))
    assert result.success
    assert "file1.txt" in result.content
    assert "file2.py" in result.content
    assert "subdir" in result.content


def test_glob(tmp_path):
    """Verifica búsqueda glob."""
    tool = Glob()
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.py").write_text("y")
    (tmp_path / "c.txt").write_text("z")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.py").write_text("w")

    result = tool.execute(pattern="*.py", root=str(tmp_path))
    assert result.success
    assert "a.py" in result.content
    assert "b.py" in result.content
    assert "c.txt" not in result.content
    assert "sub/d.py" in result.content


def test_grep(tmp_path):
    """Verifica búsqueda de texto."""
    tool = Grep()
    (tmp_path / "file1.py").write_text("def hello():\n    pass")
    (tmp_path / "file2.py").write_text("def world():\n    pass")
    (tmp_path / "file3.txt").write_text("hello world")

    result = tool.execute(pattern="def ", path=str(tmp_path))
    assert result.success
    assert "def hello" in result.content
    assert "def world" in result.content

    # Test include
    result = tool.execute(pattern="def ", path=str(tmp_path), include="*.py")
    assert result.success
    assert "file1.py" in result.content
    assert "file3.txt" not in result.content


def test_shell_command():
    """Verifica ejecución de comandos shell."""
    tool = ShellCommand()

    result = tool.execute(command="echo hola")
    assert result.success
    assert "hola" in result.content
    assert "exit code: 0" in result.content

    # Test comando fallido
    result = tool.execute(command="exit 1")
    assert not result.success
    assert "exit code: 1" in result.content


def test_shell_command_timeout():
    """Verifica timeout en shell."""
    tool = ShellCommand()
    tool.timeout = 1

    # sleep 2 debería dar timeout
    result = tool.execute(command="sleep 2", timeout=1)
    assert not result.success
    assert "timeout" in result.error.lower()


def test_dangerous_tools():
    """Verifica marcado de herramientas peligrosas."""
    registry = create_default_registry()

    write_tool = registry.get("write_file")
    assert write_tool is not None
    assert write_tool.dangerous is True

    shell_tool = registry.get("shell")
    assert shell_tool is not None
    assert shell_tool.dangerous is True

    read_tool = registry.get("read_file")
    assert read_tool is not None
    assert read_tool.dangerous is False


def test_tool_validation():
    """Verifica validación de argumentos."""
    tool = ReadFile()

    # Falta argumento requerido
    valid, error = tool.validate_args({})
    assert not valid
    assert "path" in error

    # Argumento presente
    valid, error = tool.validate_args({"path": "test.txt"})
    assert valid
    assert error is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
