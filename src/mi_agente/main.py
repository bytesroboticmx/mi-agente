"""Entry point CLI principal usando Typer."""

import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from mi_agente.agent import Agent
from mi_agente.config import Settings, get_settings, reload_settings
from mi_agente.output import AgentConsole, create_console

app = typer.Typer(
    name="mi-agente",
    help="Agente CLI inteligente para terminal Linux",
    add_completion=True,
    no_args_is_help=True,
)

console = Console()


# ─── Global options ───

def version_callback(value: bool) -> None:
    if value:
        from mi_agente import __version__
        console.print(f"mi-agente v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Modo verbose")] = False,
    model: Annotated[str | None, typer.Option("--model", help="Sobrescribe modelo")] = None,
    temp: Annotated[float | None, typer.Option("--temp", "--temperature", help="Temperatura (0-2)")] = None,
    provider: Annotated[str | None, typer.Option("--provider", help="Proveedor LLM")] = None,
    version: Annotated[bool, typer.Option("--version", "-V", callback=version_callback, is_eager=True)] = False,
) -> None:
    """Agente CLI inteligente - usa IA en tu terminal."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["override_model"] = model
    ctx.obj["override_temp"] = temp
    ctx.obj["override_provider"] = provider


def _create_agent(ctx: typer.Context) -> Agent:
    """Crea agente aplicando overrides del CLI."""
    settings = reload_settings()

    if ctx.obj.get("override_model"):
        settings.model_name = ctx.obj["override_model"]
    if ctx.obj.get("override_temp") is not None:
        settings.temperature = ctx.obj["override_temp"]
    if ctx.obj.get("override_provider"):
        settings.llm_provider = ctx.obj["override_provider"]  # type: ignore

    return Agent(
        settings=settings,
        console=create_console(verbose=ctx.obj.get("verbose", False)),
    )


# ─── Commands ───

@app.command()
def chat(
    ctx: typer.Context,
    resume: Annotated[str | None, typer.Option("--resume", "-r", help="Reanuda sesión por ID")] = None,
    new_session: Annotated[bool, typer.Option("--new", "-n", help="Inicia nueva sesión")] = False,
) -> None:
    """Inicia chat interactivo con el agente."""
    agent = _create_agent(ctx)
    if new_session:
        agent.start_session()
    elif resume:
        if not agent.resume_session(resume):
            raise typer.Exit(1)
    agent.chat()


@app.command()
def run(
    ctx: typer.Context,
    prompt: Annotated[str, typer.Argument(help="Prompt a enviar al agente")],
    stream: Annotated[bool, typer.Option("--stream/--no-stream", help="Streaming de respuesta")] = True,
) -> None:
    """Ejecuta un solo prompt y muestra la respuesta."""
    agent = _create_agent(ctx)
    agent.run(prompt, stream=stream)


@app.command()
def config(
    ctx: typer.Context,
    key: Annotated[str | None, typer.Argument(help="Clave a ver/modificar")] = None,
    value: Annotated[str | None, typer.Argument(help="Nuevo valor")] = None,
    edit: Annotated[bool, typer.Option("--edit", "-e", help="Abre .env en editor")] = False,
    show: Annotated[bool, typer.Option("--show", "-s", help="Muestra configuración actual")] = False,
) -> None:
    """Gestiona configuración del agente."""
    settings = get_settings()

    if edit:
        import subprocess
        env_path = Path(".env")
        if not env_path.exists():
            # Buscar .env.example en la raíz del proyecto
            main_file = Path(__file__)
            example = main_file.parent.parent.parent / ".env.example"
            if example.exists():
                env_path.write_text(example.read_text())
            else:
                console.print("[red].env.example no encontrado[/red]")
                return
        editor = os.environ.get("EDITOR", "code" if sys.platform != "win32" else "notepad")
        subprocess.run([editor, str(env_path)])
        reload_settings()
        console.print("[green]Configuración recargada[/green]")
        return

    if show or key is None:
        _print_config(settings)
        return

    if value is not None:
        _set_config(key, value)
        return

    _print_config_key(settings, key)


@app.command()
def config_profile(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Argument(help="Nombre del perfil")] = None,
    create: Annotated[bool, typer.Option("--create", "-c", help="Crea nuevo perfil")] = False,
    delete: Annotated[bool, typer.Option("--delete", "-d", help="Elimina perfil")] = False,
    list_all: Annotated[bool, typer.Option("--list", "-l", help="Lista perfiles")] = False,
) -> None:
    """Gestiona perfiles de configuración."""
    profiles_dir = Path.home() / ".mi-agente" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    if list_all or (not create and not delete):
        profiles = list(profiles_dir.glob("*.env"))
        if profiles:
            console.print("**Perfiles disponibles:**")
            for p in profiles:
                console.print(f"  • {p.stem}")
        else:
            console.print("[yellow]No hay perfiles guardados[/yellow]")
        return

    profile_path = profiles_dir / f"{name}.env"

    if create:
        if profile_path.exists():
            console.print(f"[yellow]Perfil '{name}' ya existe[/yellow]")
            return
        # Copiar .env actual como perfil
        current_env = Path(".env")
        if current_env.exists():
            profile_path.write_text(current_env.read_text())
            console.print(f"[green]Perfil '{name}' creado[/green]")
        else:
            console.print("[red]No hay .env actual para copiar[/red]")
        return

    if delete:
        if profile_path.exists():
            profile_path.unlink()
            console.print(f"[green]Perfil '{name}' eliminado[/green]")
        else:
            console.print(f"[red]Perfil '{name}' no encontrado[/red]")
        return


@app.command()
def config_use(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Nombre del perfil a usar")],
) -> None:
    """Cambia al perfil especificado (copia a .env)."""
    profiles_dir = Path.home() / ".mi-agente" / "profiles"
    profile_path = profiles_dir / f"{name}.env"
    
    if not profile_path.exists():
        console.print(f"[red]Perfil '{name}' no encontrado[/red]")
        return
    
    env_path = Path(".env")
    env_path.write_text(profile_path.read_text())
    reload_settings()
    console.print(f"[green]Perfil '{name}' activado[/green]")


@app.command()
def tools(
    ctx: typer.Context,
    list_all: Annotated[bool, typer.Option("--list", "-l", help="Lista todas las herramientas")] = True,
) -> None:
    """Lista herramientas disponibles."""
    from mi_agente.tools import create_default_registry
    from mi_agente.output import create_console
    
    registry = create_default_registry()
    console = create_console(verbose=ctx.obj.get("verbose", False))
    console.print_md("**Herramientas disponibles:**")
    for tool in registry.list():
        danger = " ⚠️" if tool.dangerous else ""
        console.print(f"  • `{tool.name}`{danger}: {tool.description}")


@app.command()
def sessions(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", "-l", help="Límite de sesiones")] = 10,
) -> None:
    """Lista sesiones de conversación guardadas."""
    agent = _create_agent(ctx)
    agent._print_sessions()


@app.command()
def init(
    ctx: typer.Context,
    force: Annotated[bool, typer.Option("--force", "-f", help="Sobrescribe archivos existentes")] = False,
) -> None:
    """Inicializa proyecto en directorio actual."""
    env_path = Path(".env")
    if env_path.exists() and not force:
        console.print("[yellow].env ya existe. Usa --force para sobrescribir[/yellow]")
        return

    # Buscar .env.example en la raíz del proyecto (2 niveles arriba de main.py)
    main_file = Path(__file__)
    example = main_file.parent.parent.parent / ".env.example"
    if example.exists():
        env_path.write_text(example.read_text())
        console.print("[green]Creado .env desde .env.example[/green]")
    else:
        console.print("[red].env.example no encontrado[/red]")


@app.command()
def completion(
    shell: Annotated[str, typer.Argument(help="Shell: bash, zsh, fish")],
) -> None:
    """Genera script de autocompletado para tu shell."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "mi_agente.main", "--install-completion"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        console.print(result.stdout)
    else:
        console.print(f"[red]Error: {result.stderr}[/red]")


@app.command()
def install_completion(
    shell: Annotated[str, typer.Argument(help="Shell: bash, zsh, fish")],
    path: Annotated[str | None, typer.Option("--path", help="Ruta donde instalar")] = None,
) -> None:
    """Instala autocompletado en el shell."""
    import subprocess
    cmd = [sys.executable, "-m", "mi_agente.main", "--install-completion"]
    if shell:
        cmd.extend(["--shell", shell])
    if path:
        cmd.extend(["--path", path])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        console.print("[green]Autocompletado instalado[/green]")
        console.print(result.stdout)
    else:
        console.print(f"[red]Error: {result.stderr}[/red]")


# ─── Helpers ───

def _print_config(settings: Settings) -> None:
    c = create_console()
    c.print_md(f"""
**Configuración actual:**
- **Proveedor**: `{settings.llm_provider}`
- **Modelo**: `{settings.model_name}`
- **Temperatura**: `{settings.temperature}`
- **Max tokens**: `{settings.max_tokens}`
- **System prompt**: `{settings.system_prompt[:80]}...`
- **Directorio trabajo**: `{settings.working_dir}`
- **Tools peligrosas requieren confirmación**: `{settings.confirm_dangerous_tools}`
- **Memoria habilitada**: `{settings.memory_enabled}`
- **Tema**: `{settings.theme}`
    """)


def _print_config_key(settings: Settings, key: str) -> None:
    if hasattr(settings, key):
        val = getattr(settings, key)
        if key == "api_key":
            val = "***" if val else "(vacío)"
        console.print(f"{key} = {val}")
    else:
        console.print(f"[red]Clave desconocida: {key}[/red]")


def _set_config(key: str, value: str) -> None:
    env_path = Path(".env")
    lines = []
    if env_path.exists():
        lines = env_path.read_text().splitlines()

    # Actualizar o añadir
    found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            found = True
            break

    if not found:
        lines.append(f"{key}={value}")

    env_path.write_text("\n".join(lines) + "\n")
    reload_settings()
    console.print(f"[green]Configurado {key}={value}[/green]")


# ─── Version ───

try:
    from mi_agente import __version__
except ImportError:
    __version__ = "0.1.0"


if __name__ == "__main__":
    app()