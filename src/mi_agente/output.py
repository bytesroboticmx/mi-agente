"""Formato de salida bonito usando Rich."""

import atexit
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.theme import Theme

from mi_agente.config import get_settings

if TYPE_CHECKING:
    from rich.status import Status

# ─── Temas de color ───

DEFAULT_THEME = Theme(
    {
        "info": "bold cyan",
        "warning": "bold yellow",
        "error": "bold red",
        "success": "bold green",
        "dim": "dim",
        "panel.system": "dim cyan",
        "panel.user": "green",
        "panel.assistant": "blue",
        "panel.tool": "yellow",
        "panel.tool_error": "red",
        "prompt": "bold green",
    }
)

DARK_THEME = Theme(
    {
        "info": "bold bright_cyan",
        "warning": "bold bright_yellow",
        "error": "bold bright_red",
        "success": "bold bright_green",
        "dim": "dim",
        "panel.system": "bright_black",
        "panel.user": "bright_green",
        "panel.assistant": "bright_blue",
        "panel.tool": "bright_yellow",
        "panel.tool_error": "bright_red",
        "prompt": "bold bright_green",
    }
)

LIGHT_THEME = Theme(
    {
        "info": "bold blue",
        "warning": "bold orange3",
        "error": "bold red",
        "success": "bold green",
        "dim": "dim",
        "panel.system": "grey50",
        "panel.user": "green",
        "panel.assistant": "blue",
        "panel.tool": "yellow",
        "panel.tool_error": "red",
        "prompt": "bold green",
    }
)

THEMES = {
    "default": DEFAULT_THEME,
    "dark": DARK_THEME,
    "light": LIGHT_THEME,
}


# ─── Historial de input ───


class InputHistory:
    """Gestiona historial de comandos persistente."""

    def __init__(self, history_file: Path):
        self.history_file = history_file
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self._history: list[str] = []
        self._load()

    def _load(self) -> None:
        if self.history_file.exists():
            try:
                self._history = self.history_file.read_text().splitlines()[-1000:]
            except Exception:
                self._history = []

    def _save(self) -> None:
        try:
            self.history_file.write_text("\n".join(self._history[-1000:]) + "\n")
        except Exception:
            pass

    def add(self, entry: str) -> None:
        if entry and (not self._history or self._history[-1] != entry):
            self._history.append(entry)

    def get(self) -> list[str]:
        return self._history.copy()


# ─── Consola principal ───


class AgentConsole:
    """Wrapper sobre Rich Console para output consistente."""

    def __init__(self, verbose: bool = False, theme_name: str = "default"):
        theme = THEMES.get(theme_name, DEFAULT_THEME)
        self.console = Console(theme=theme)
        self.verbose = verbose
        self._live: Live | None = None
        self._streaming_content = ""

        # Historial
        settings = get_settings()
        self.history = InputHistory(settings.memory_dir / "input_history.txt")
        atexit.register(self._save_history)

    def _save_history(self) -> None:
        self.history._save()

    # ─── Basic output ───

    def print(self, *args: Any, **kwargs: Any) -> None:
        self.console.print(*args, **kwargs)

    def print_md(self, markdown: str) -> None:
        """Imprime markdown renderizado."""
        self.console.print(Markdown(markdown))

    def print_code(self, code: str, language: str = "python") -> None:
        """Imprime código con syntax highlighting."""
        self.console.print(Syntax(code, language, theme="monokai", line_numbers=True))

    def print_panel(
        self,
        content: str,
        title: str = "",
        style: str = "panel.assistant",
        border_style: str = "panel.assistant",
    ) -> None:
        """Imprime un panel con título."""
        panel = Panel(
            content,
            title=title,
            title_align="left",
            style=style,
            border_style=border_style,
            padding=(0, 1),
        )
        self.console.print(panel)

    def print_system(self, content: str) -> None:
        self.print_panel(
            content, title="🤖 System", style="panel.system", border_style="panel.system"
        )

    def print_user(self, content: str) -> None:
        self.print_panel(content, title="👤 User", style="panel.user", border_style="panel.user")

    def print_assistant(self, content: str) -> None:
        self.print_panel(
            content, title="💬 Assistant", style="panel.assistant", border_style="panel.assistant"
        )

    def print_tool_call(self, name: str, args: dict) -> None:
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        self.print_panel(
            f"[bold]{name}[/bold]({args_str})",
            title="🔧 Tool Call",
            style="panel.tool",
            border_style="panel.tool",
        )

    def print_tool_result(self, name: str, result: str, is_error: bool = False) -> None:
        style = "panel.tool_error" if is_error else "panel.tool"
        border = "panel.tool_error" if is_error else "panel.tool"
        title = f"🔧 Tool Result: {name}" + (" (error)" if is_error else "")
        self.print_panel(result, title=title, style=style, border_style=border)

    def print_error(self, message: str) -> None:
        self.console.print(f"[error]Error:[/error] {message}")

    def print_warning(self, message: str) -> None:
        self.console.print(f"[warning]Advertencia:[/warning] {message}")

    def print_info(self, message: str) -> None:
        self.console.print(f"[info]Info:[/info] {message}")

    def print_success(self, message: str) -> None:
        self.console.print(f"[success]✓[/success] {message}")

    def print_spinner(self, message: str = "Pensando...") -> "SpinnerContext":
        """Retorna context manager con spinner."""
        return SpinnerContext(self, message)

    # ─── Streaming ───

    def start_streaming(self, title: str = "Assistant") -> None:
        """Inicia la visualización en vivo de la respuesta."""
        self._streaming_content = ""
        self._live = Live(
            Markdown(""),
            console=self.console,
            refresh_per_second=15,
            transient=False,
        )
        self._live.__enter__()

    def add_stream_chunk(self, chunk: str) -> None:
        """Añade chunk al streaming actual."""
        self._streaming_content += chunk
        if self._live:
            self._live.update(Markdown(self._streaming_content))

    def stop_streaming(self) -> None:
        """Finaliza la visualización en vivo."""
        if self._live is not None:
            self._live.__exit__(None, None, None)
            self._live = None

    def end_streaming(self) -> str:
        """Finaliza streaming y retorna contenido completo."""
        self.stop_streaming()
        content = self._streaming_content
        self._streaming_content = ""
        return content

    # ─── Input con historial ───

    def input(self, prompt: str = "") -> str:
        """Input con soporte de historial (flechas arriba/abajo)."""
        try:
            # Usar readline si está disponible para historial
            import readline

            readline.set_history_length(1000)
            for item in self.history.get():
                readline.add_history(item)
        except ImportError:
            pass

        result = self.console.input(prompt)
        if result:
            self.history.add(result)
        return result


class SpinnerContext:
    """Context manager para spinner de carga."""

    def __init__(self, console: AgentConsole, message: str):
        self.console = console
        self.message = message
        self._spinner: Status | None = None

    def __enter__(self) -> "SpinnerContext":
        self._spinner = self.console.console.status(self.message, spinner="dots")
        self._spinner.__enter__()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        if self._spinner:
            self._spinner.__exit__(exc_type, exc_val, exc_tb)  # type: ignore[arg-type]


# ─── Helpers ───


def create_console(verbose: bool = False, theme: str = "default") -> AgentConsole:
    """Factory para crear consola."""
    return AgentConsole(verbose=verbose, theme_name=theme)


def confirm(prompt: str, default: bool = True) -> bool:
    """Pide confirmación al usuario."""
    console = Console()
    return console.input(f"[yellow]{prompt} [y/N]: [/yellow]").lower() in ("y", "yes", "s", "si")


def select_option(prompt: str, options: list[str], default: int = 0) -> int:
    """Menú de selección simple."""
    console = Console()
    console.print(f"\n[bold]{prompt}[/bold]")
    for i, opt in enumerate(options):
        prefix = "▶" if i == default else " "
        console.print(f"  {prefix} {i + 1}. {opt}")
    while True:
        try:
            choice = console.input(f"\nSelecciona [1-{len(options)}] (default {default + 1}): ")
            if not choice:
                return default
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        console.print("[red]Opción inválida[/red]")


def get_available_themes() -> list[str]:
    """Retorna lista de temas disponibles."""
    return list(THEMES.keys())
