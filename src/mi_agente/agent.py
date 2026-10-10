"""Núcleo del agente: loop principal y manejo de herramientas."""

from dataclasses import dataclass, field
from typing import Any, Optional

from mi_agente.config import Settings, get_settings
from mi_agente.llm import LLMResponse, Message, ToolSpec, create_llm_client
from mi_agente.memory import ContextManager, ConversationStore
from mi_agente.output import AgentConsole, create_console
from mi_agente.tools import ToolRegistry, ToolResult, create_default_registry


@dataclass
class Agent:
    """Agente principal con loop de conversación."""

    settings: Settings = field(default_factory=get_settings)
    console: AgentConsole = field(default_factory=create_console)
    tools: ToolRegistry = field(default_factory=create_default_registry)
    system_prompt: str | None = None
    messages: list[Message] = field(default_factory=list)
    max_context_tokens: int = 100000
    session_id: Optional[str] = None
    _llm: Optional[Any] = field(default=None, init=False, repr=False)
    _memory: Optional[ConversationStore] = field(default=None, init=False, repr=False)
    _context_manager: Optional[ContextManager] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.system_prompt is None:
            self.system_prompt = self.settings.system_prompt
        if self.settings.memory_enabled:
            self._memory = ConversationStore()
            self._context_manager = ContextManager()
        
        # Recreate console with theme
        self.console = create_console(theme=self.settings.theme)
        
        self._init_messages()

    @property
    def llm(self):
        """Lazy initialization of LLM client."""
        if self._llm is None:
            self._llm = create_llm_client(self.settings)
        return self._llm

    @property
    def memory(self) -> Optional[ConversationStore]:
        return self._memory

    @property
    def context_manager(self) -> Optional[ContextManager]:
        return self._context_manager

    def _init_messages(self) -> None:
        """Inicializa mensajes con system prompt."""
        self.messages = [Message(role="system", content=self.system_prompt)]

    def add_user_message(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))
        self._maybe_save()

    def add_assistant_message(self, content: str | None, tool_calls: list[dict] | None = None) -> None:
        self.messages.append(Message(role="assistant", content=content, tool_calls=tool_calls))
        self._maybe_save()

    def add_tool_result(self, tool_call_id: str, name: str, result: str) -> None:
        self.messages.append(Message(role="tool", content=result, tool_call_id=tool_call_id, name=name))
        self._maybe_save()

    def _maybe_save(self) -> None:
        """Guarda en memoria si está habilitado y hay sesión activa."""
        if self.memory and self.session_id:
            self.memory.save_messages(self.session_id, self.messages)

    def get_tool_specs(self) -> list[ToolSpec]:
        """Convierte tools registradas a ToolSpec para el LLM."""
        specs = []
        for tool in self.tools.list():
            specs.append(ToolSpec(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            ))
        return specs

    def run(self, user_input: str, stream: bool = True) -> str:
        """Ejecuta un turno completo: user -> assistant -> tools -> assistant."""
        self.add_user_message(user_input)
        self.console.print_user(user_input)

        # Primera respuesta del LLM
        response = self._call_llm(stream=stream)
        final_content = self._handle_response(response, stream)

        # Si hubo tool calls, ejecutar y obtener respuesta final
        while response.tool_calls:
            for tc in response.tool_calls:
                self._execute_tool_call(tc)

            # Segunda llamada al LLM con resultados
            response = self._call_llm(stream=stream)
            final_content = self._handle_response(response, stream)

        return final_content or ""

    def _call_llm(self, stream: bool) -> LLMResponse:
        """Llama al LLM con mensajes y tools actuales."""
        tool_specs = self.get_tool_specs()

        # Truncar contexto si es necesario (con resumen inteligente)
        self._truncate_context()

        response = self.llm.complete(
            messages=self.messages,
            tools=tool_specs if tool_specs else None,
            stream=stream,
        )

        if stream:
            # Consumir stream y acumular
            chunks = []
            for chunk in response:
                chunks.append(chunk)
            # Reconstruir respuesta final
            content_parts = [c.content for c in chunks if c.content]
            tool_calls = next((c.tool_calls for c in chunks if c.tool_calls), None)
            finish_reason = next((c.finish_reason for c in chunks if c.finish_reason), None)
            model = next((c.model for c in chunks if c.model), None)

            return LLMResponse(
                content="".join(content_parts) if content_parts else None,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                model=model,
            )
        return response

    def _handle_response(self, response: LLMResponse, stream: bool) -> str | None:
        """Procesa respuesta del LLM (imprime, guarda en historial)."""
        if stream and self.console._live:
            # El streaming ya se mostró en vivo
            final_content = self.console.end_streaming()
            if final_content:
                self.add_assistant_message(final_content, response.tool_calls)
                return final_content
        else:
            if response.content:
                self.console.print_assistant(response.content)
            if response.tool_calls:
                for tc in response.tool_calls:
                    self.console.print_tool_call(
                        tc["function"]["name"],
                        tc["function"].get("arguments", {}),
                    )
            self.add_assistant_message(response.content, response.tool_calls)
            return response.content
        return None

    def _execute_tool_call(self, tool_call: dict) -> None:
        """Ejecuta una tool call y añade resultado al historial."""
        func = tool_call["function"]
        name = func["name"]
        args_str = func.get("arguments", "{}")

        import json
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}

        self.console.print_tool_call(name, args)

        result = self.tools.execute(name, args, confirm=self.settings.confirm_dangerous_tools)

        if result.success:
            self.console.print_tool_result(name, result.content)
        else:
            self.console.print_tool_result(name, result.error or "Error desconocido", is_error=True)

        self.add_tool_result(tool_call["id"], name, result.content if result.success else result.error or "Error")

    def _truncate_context(self) -> None:
        """Trunca mensajes si exceden límite de tokens con resumen inteligente."""
        if not self.context_manager:
            # Fallback simple
            if len(self.messages) > 21:
                system = self.messages[0]
                self.messages = [system] + self.messages[-20:]
            return

        # Verificar si necesita resumen
        if self.context_manager.needs_summary(self.messages):
            self.console.print_info("Contexto largo detectado, generando resumen...")
            summary = self._generate_summary()
            if summary:
                self.messages = self.context_manager.truncate_with_summary(
                    self.messages, summary, keep_last=10
                )
                self.console.print_success("Contexto resumido")
                self._maybe_save()

    def _generate_summary(self) -> str | None:
        """Genera resumen de la conversación usando el LLM."""
        if not self.context_manager or len(self.messages) < 5:
            return None

        try:
            summary_prompt = self.context_manager.create_summary_prompt(self.messages)
            # Usar una llamada simple sin tools para el resumen
            response = self.llm.complete(
                messages=[Message(role="user", content=summary_prompt)],
                tools=None,
                stream=False,
            )
            return response.content
        except Exception as e:
            self.console.print_warning(f"No se pudo generar resumen: {e}")
            return None

    def start_session(self, name: Optional[str] = None) -> str:
        """Inicia una nueva sesión de conversación."""
        if not self.memory:
            return ""

        session = self.memory.create_session(
            name=name,
            model=self.settings.model_name,
            provider=self.settings.llm_provider,
            system_prompt=self.system_prompt or "",
        )
        self.session_id = session.id
        self.messages = [Message(role="system", content=self.system_prompt)]
        self.console.print_success(f"Sesión iniciada: {session.id}" + (f" ({name})" if name else ""))
        return session.id

    def resume_session(self, session_id: str) -> bool:
        """Reanuda una sesión existente."""
        if not self.memory:
            return False

        session = self.memory.load_session(session_id)
        if not session:
            self.console.print_error(f"Sesión no encontrada: {session_id}")
            return False

        self.session_id = session.id
        self.messages = self.memory.load_messages(session_id)
        if not self.messages or self.messages[0].role != "system":
            self.messages.insert(0, Message(role="system", content=self.system_prompt))
        self.console.print_success(f"Sesión reanudada: {session_id}" + (f" ({session.name})" if session.name else ""))
        return True

    def list_sessions(self, limit: int = 10) -> list:
        """Lista sesiones guardadas."""
        if not self.memory:
            return []
        return self.memory.list_sessions(limit)

    def chat(self) -> None:
        """Loop interactivo de chat."""
        self.console.print_info("Modo chat interactivo. Escribe 'exit' o 'quit' para salir.")
        self.console.print_info("Comandos: /help, /tools, /clear, /config, /sessions, /new, /resume <id>")

        while True:
            try:
                user_input = self.console.input("\n[prompt]▶[/prompt] ").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]Saliendo...[/dim]")
                break

            if not user_input:
                continue

            # Comandos internos
            if user_input.lower() in ("exit", "quit", "/q"):
                break
            elif user_input == "/help":
                self._print_help()
                continue
            elif user_input == "/tools":
                self._print_tools()
                continue
            elif user_input == "/clear":
                self._init_messages()
                self.console.print_success("Historial limpiado")
                continue
            elif user_input == "/config":
                self._print_config()
                continue
            elif user_input == "/sessions":
                self._print_sessions()
                continue
            elif user_input == "/new":
                self.start_session()
                continue
            elif user_input.startswith("/resume "):
                session_id = user_input.split(" ", 1)[1].strip()
                self.resume_session(session_id)
                continue

            self.run(user_input, stream=True)

    def _print_help(self) -> None:
        help_text = """
**Comandos disponibles:**
- `/help` - Muestra esta ayuda
- `/tools` - Lista herramientas disponibles
- `/clear` - Limpia historial de conversación
- `/config` - Muestra configuración actual
- `/sessions` - Lista sesiones guardadas
- `/new` - Nueva sesión
- `/resume <id>` - Reanuda sesión por ID
- `exit` / `quit` / `/q` - Sale del chat
        """
        self.console.print_md(help_text)

    def _print_tools(self) -> None:
        self.console.print_md("**Herramientas disponibles:**")
        for tool in self.tools.list():
            danger = " ⚠️" if tool.dangerous else ""
            self.console.print(f"  • `{tool.name}`{danger}: {tool.description}")

    def _print_config(self) -> None:
        self.console.print_md(f"""
**Configuración actual:**
- Proveedor: `{self.settings.llm_provider}`
- Modelo: `{self.settings.model_name}`
- Temperatura: `{self.settings.temperature}`
- Max tokens: `{self.settings.max_tokens}`
- Directorio: `{self.settings.working_dir}`
- Memoria: `{'activada' if self.settings.memory_enabled else 'desactivada'}`
        """)

    def _print_sessions(self) -> None:
        sessions = self.list_sessions()
        if not sessions:
            self.console.print_info("No hay sesiones guardadas")
            return

        self.console.print_md("**Sesiones guardadas:**")
        for s in sessions:
            name = f" ({s.name})" if s.name else ""
            current = " ← actual" if s.id == self.session_id else ""
            self.console.print(
                f"  • `{s.id}`{name}{current} - {s.message_count} msgs - "
                f"{s.updated_at.strftime('%Y-%m-%d %H:%M')}"
            )