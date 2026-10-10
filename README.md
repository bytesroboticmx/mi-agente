# mi-agente

Agente CLI inteligente para terminal Linux. Conversa con LLMs, usa herramientas (archivos, shell, búsqueda) y mantiene contexto.

## Características

- 🤖 **Multi-proveedor**: OpenAI, Anthropic, Ollama (local)
- 🔧 **Herramientas integradas**: Leer/escribir archivos, listar directorios, glob, grep, shell
- 💬 **Chat interactivo** con streaming de respuestas
- 🧠 **Memoria persistente** (SQLite) con sesiones y resumen automático
- ⚙️ **Configuración simple** vía `.env` + perfiles
- 🎨 **Output bonito** con Rich (colores, markdown, paneles, syntax highlighting)
- 🔒 **Seguridad**: Confirmación para herramientas peligrosas
- ⌨️ **Autocompletado** para bash/zsh/fish
- 🎨 **Temas**: default, dark, light

## Instalación

```bash
# Clonar y entrar
cd mi-agente

# Crear venv e instalar
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Copiar config y editar
cp .env.example .env
# Edita .env con tu API key

# Instalar autocompletado (opcional)
mi-agente install-completion bash  # o zsh, fish
```

## Uso rápido

```bash
# Chat interactivo
mi-agente chat

# Nueva sesión con nombre
mi-agente chat --new

# Reanudar sesión
mi-agente chat -r <session-id>

# Un solo prompt
mi-agente run "Lista archivos en el directorio actual"

# Ver herramientas
mi-agente tools

# Ver sesiones guardadas
mi-agente sessions

# Configuración
mi-agente config --show
mi-agente config --edit        # Abre .env en editor
mi-agente config THEME dark    # Cambia tema
mi-agente config-profile --create myprofile  # Guarda perfil
mi-agente config-use myprofile               # Usa perfil
```

## Configuración (`.env`)

```bash
# Proveedor: openai | anthropic | ollama
LLM_PROVIDER=openai

# Modelo
MODEL_NAME=gpt-4o-mini

# API Key (requerida para OpenAI/Anthropic)
API_KEY=sk-tu-key

# Para Ollama local
# BASE_URL=http://localhost:11434/v1

# Parámetros
TEMPERATURE=0.7
MAX_TOKENS=4096

# System prompt personalizado
SYSTEM_PROMPT=Eres un asistente experto en Linux y desarrollo...

# Tema: default | dark | light
THEME=default
```

## Comandos en chat

```
/help       - Ayuda
/tools      - Lista herramientas
/clear      - Limpia historial
/config     - Ver config
/sessions   - Lista sesiones guardadas
/new        - Nueva sesión
/resume <id> - Reanuda sesión
exit/quit   - Salir
```

## Herramientas disponibles

| Herramienta | Descripción | Peligrosa |
|-------------|-------------|-----------|
| `read_file` | Lee archivo | No |
| `write_file` | Escribe archivo | ✅ Sí |
| `list_dir` | Lista directorio | No |
| `glob` | Busca archivos por patrón | No |
| `grep` | Busca texto en archivos | No |
| `shell` | Ejecuta comando shell | ✅ Sí |

## Perfiles de configuración

```bash
# Crear perfil desde .env actual
mi-agente config-profile myprofile --create

# Listar perfiles
mi-agente config-profile --list

# Cambiar a perfil
mi-agente config-use myprofile

# Eliminar perfil
mi-agente config-profile myprofile --delete
```

Los perfiles se guardan en `~/.mi-agente/profiles/`.

## Memoria y sesiones

- Las conversaciones se guardan automáticamente en SQLite (`~/.mi-agente/memory/conversations.db`)
- Cada sesión tiene ID corto (8 chars) y nombre opcional
- Resumen automático cuando el contexto supera ~6000 tokens
- Mantiene: system prompt + resumen + últimos 10 mensajes

## Ejemplos

```bash
# Leer y analizar código
mi-agente run "Explica qué hace el archivo src/main.py"

# Crear archivo
mi-agente run "Crea un script Python que haga backup de .env"

# Buscar en código
mi-agente run "Busca todas las funciones async en el proyecto"

# Ejecutar comandos
mi-agente run "Ejecuta pytest y dime si pasan los tests"

# Usar Ollama local
mi-agente config LLM_PROVIDER ollama
mi-agente config MODEL_NAME llama3.1
mi-agente chat
```

## Desarrollo

```bash
# Lint
ruff check .

# Format
ruff format .

# Type check
mypy src

# Tests
pytest
```

## Estructura

```
src/mi_agente/
├── main.py          # CLI entry point (Typer)
├── config.py        # Configuración (Pydantic + python-dotenv)
├── agent.py         # Núcleo del agente
├── llm.py           # Interfaz LLM base
├── llm_openai.py    # Cliente OpenAI
├── llm_anthropic.py # Cliente Anthropic
├── llm_ollama.py    # Cliente Ollama
├── output.py        # Formato salida (Rich + temas)
├── memory.py        # Memoria (SQLite + resumen)
└── tools/
    ├── base.py      # Base Tool + Registry
    ├── filesystem.py # read, write, list, glob, grep
    └── shell.py     # shell command
```

## Tests

```bash
pytest tests/ -v
```

26 tests cubren: config, tools (filesystem + shell), memory (SQLite + context manager).

## CI/CD

GitHub Actions workflow en `.github/workflows/ci.yml`:
- Lint (ruff), format check, type check (mypy)
- Tests en Python 3.11, 3.12, 3.13
- Build y verificación de package
- Release automático a PyPI en tags

## Roadmap

- [ ] RAG simple (embeddings locales con sentence-transformers)
- [ ] Packaging binario (PyInstaller)
- [ ] Plugin system para tools custom
- [ ] Web search / fetch tools

## Licencia

MIT