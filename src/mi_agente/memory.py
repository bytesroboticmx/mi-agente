"""Memoria persistente del agente usando SQLite."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mi_agente.config import get_settings
from mi_agente.llm import Message


@dataclass
class ConversationSession:
    """Sesión de conversación guardada."""
    id: str
    name: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int
    model: str
    provider: str
    system_prompt: str
    metadata: dict = field(default_factory=dict)


class ConversationStore:
    """Almacena y recupera conversaciones en SQLite."""

    def __init__(self, db_path: Optional[Path] = None):
        settings = get_settings()
        self.db_path = db_path or (settings.memory_dir / "conversations.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Inicializa la base de datos."""
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    message_count INTEGER DEFAULT 0,
                    model TEXT,
                    provider TEXT,
                    system_prompt TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    tool_calls TEXT,
                    tool_call_id TEXT,
                    name TEXT,
                    tokens INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session 
                ON messages(session_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_updated 
                ON sessions(updated_at DESC)
            """)

    @contextmanager
    def _conn(self):
        """Context manager para conexión SQLite."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_session(
        self,
        name: Optional[str] = None,
        model: str = "",
        provider: str = "",
        system_prompt: str = "",
        metadata: Optional[dict] = None,
    ) -> ConversationSession:
        """Crea una nueva sesión."""
        session_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO sessions 
                   (id, name, created_at, updated_at, model, provider, system_prompt, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, name, now, now, model, provider, system_prompt, json.dumps(metadata or {}))
            )

        return ConversationSession(
            id=session_id,
            name=name,
            created_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
            message_count=0,
            model=model,
            provider=provider,
            system_prompt=system_prompt,
            metadata=metadata or {},
        )

    def save_messages(self, session_id: str, messages: list[Message]) -> None:
        """Guarda mensajes de una sesión (reemplaza todos)."""
        with self._conn() as conn:
            # Borrar mensajes existentes
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))

            # Insertar nuevos
            for msg in messages:
                conn.execute(
                    """INSERT INTO messages 
                       (session_id, role, content, tool_calls, tool_call_id, name, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        msg.role,
                        msg.content,
                        json.dumps(msg.tool_calls) if msg.tool_calls else None,
                        msg.tool_call_id,
                        msg.name,
                        datetime.now().isoformat(),
                    )
                )

            # Actualizar contador y timestamp
            conn.execute(
                "UPDATE sessions SET message_count = ?, updated_at = ? WHERE id = ?",
                (len(messages), datetime.now().isoformat(), session_id),
            )

    def load_session(self, session_id: str) -> Optional[ConversationSession]:
        """Carga una sesión por ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()

            if not row:
                return None

            return ConversationSession(
                id=row["id"],
                name=row["name"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                message_count=row["message_count"],
                model=row["model"] or "",
                provider=row["provider"] or "",
                system_prompt=row["system_prompt"] or "",
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )

    def load_messages(self, session_id: str) -> list[Message]:
        """Carga todos los mensajes de una sesión."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()

            messages = []
            for row in rows:
                msg = Message(
                    role=row["role"],
                    content=row["content"],
                    tool_calls=json.loads(row["tool_calls"]) if row["tool_calls"] else None,
                    tool_call_id=row["tool_call_id"],
                    name=row["name"],
                )
                messages.append(msg)
            return messages

    def list_sessions(self, limit: int = 20) -> list[ConversationSession]:
        """Lista sesiones recientes."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()

            return [
                ConversationSession(
                    id=row["id"],
                    name=row["name"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    message_count=row["message_count"],
                    model=row["model"] or "",
                    provider=row["provider"] or "",
                    system_prompt=row["system_prompt"] or "",
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )
                for row in rows
            ]

    def delete_session(self, session_id: str) -> bool:
        """Elimina una sesión."""
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cursor.rowcount > 0

    def update_session_metadata(self, session_id: str, metadata: dict) -> None:
        """Actualiza metadata de sesión."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET metadata = ?, updated_at = ? WHERE id = ?",
                (json.dumps(metadata), datetime.now().isoformat(), session_id),
            )


class ContextManager:
    """Gestiona contexto y resúmenes para conversaciones largas."""

    def __init__(self, max_tokens: int = 8000, summary_threshold: int = 6000):
        self.max_tokens = max_tokens
        self.summary_threshold = summary_threshold

    def estimate_tokens(self, messages: list[Message]) -> int:
        """Estima tokens totales (aproximación)."""
        total = 0
        for msg in messages:
            if msg.content:
                total += len(msg.content) // 4
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total += len(str(tc)) // 4
        return total

    def needs_summary(self, messages: list[Message]) -> bool:
        """Verifica si necesita resumen."""
        return self.estimate_tokens(messages) > self.summary_threshold

    def create_summary_prompt(self, messages: list[Message]) -> str:
        """Crea prompt para resumir conversación."""
        conversation = "\n".join(
            f"{m.role}: {m.content or ''}" for m in messages if m.content
        )
        return (
            "Resume la siguiente conversación manteniendo los puntos clave, "
            "decisiones importantes, y cualquier información relevante para "
            "continuar la conversación:\n\n" + conversation
        )

    def truncate_with_summary(
        self,
        messages: list[Message],
        summary: str,
        keep_last: int = 10,
    ) -> list[Message]:
        """Trunca mensajes manteniendo resumen + últimos N."""
        if len(messages) <= keep_last + 1:  # +1 por system
            return messages

        system_msg = messages[0] if messages and messages[0].role == "system" else None
        recent = messages[-(keep_last):]

        summary_msg = Message(
            role="system",
            content=f"[Resumen de conversación previa]\n{summary}"
        )

        result = []
        if system_msg:
            result.append(system_msg)
        result.append(summary_msg)
        result.extend(recent)
        return result