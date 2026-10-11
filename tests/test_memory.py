"""Tests para memoria y contexto."""

import pytest

from mi_agente.llm import Message
from mi_agente.memory import ContextManager, ConversationStore


def test_conversation_store(tmp_path):
    """Verifica almacenamiento de conversaciones."""
    db_path = tmp_path / "test.db"
    store = ConversationStore(db_path)

    # Crear sesión
    session = store.create_session(
        name="Test Session",
        model="gpt-4o",
        provider="openai",
        system_prompt="Eres un asistente",
    )
    assert session.id
    assert session.name == "Test Session"
    assert session.message_count == 0

    # Guardar mensajes
    messages = [
        Message(role="system", content="Eres un asistente"),
        Message(role="user", content="Hola"),
        Message(role="assistant", content="¡Hola!"),
    ]
    store.save_messages(session.id, messages)

    # Cargar sesión
    loaded_session = store.load_session(session.id)
    assert loaded_session is not None
    assert loaded_session.message_count == 3

    # Cargar mensajes
    loaded_messages = store.load_messages(session.id)
    assert len(loaded_messages) == 3
    assert loaded_messages[0].role == "system"
    assert loaded_messages[1].content == "Hola"
    assert loaded_messages[2].content == "¡Hola!"

    # Listar sesiones
    sessions = store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].id == session.id


def test_conversation_store_delete(tmp_path):
    """Verifica eliminación de sesión."""
    db_path = tmp_path / "test.db"
    store = ConversationStore(db_path)

    session = store.create_session(name="To Delete")
    assert store.load_session(session.id) is not None

    deleted = store.delete_session(session.id)
    assert deleted is True
    assert store.load_session(session.id) is None

    # Eliminar inexistente
    deleted = store.delete_session("nonexistent")
    assert deleted is False


def test_context_manager_estimate_tokens():
    """Verifica estimación de tokens."""
    cm = ContextManager()

    messages = [
        Message(role="system", content="System prompt"),
        Message(role="user", content="Hola" * 100),  # ~400 chars = ~100 tokens
        Message(role="assistant", content="Respuesta" * 100),
    ]

    tokens = cm.estimate_tokens(messages)
    assert tokens > 0
    assert tokens < 1000  # Debería ser aproximado


def test_context_manager_needs_summary():
    """Verifica detección de necesidad de resumen."""
    cm = ContextManager(summary_threshold=100)

    # Pocos mensajes - no necesita resumen
    messages = [Message(role="system", content="S"), Message(role="user", content="H")]
    assert not cm.needs_summary(messages)

    # Muchos mensajes - necesita resumen
    messages = [
        Message(role="system", content="S"),
        *[Message(role="user", content="H" * 100) for _ in range(10)],
    ]
    assert cm.needs_summary(messages)


def test_context_manager_truncate_with_summary():
    """Verifica truncado con resumen."""
    cm = ContextManager()

    messages = [
        Message(role="system", content="System"),
        *[Message(role="user", content=f"Msg {i}") for i in range(20)],
    ]

    truncated = cm.truncate_with_summary(messages, "Resumen de conversación", keep_last=5)

    # Debe mantener system + resumen + últimos 5
    assert len(truncated) == 7  # 1 system + 1 resumen + 5 últimos
    assert truncated[0].role == "system"
    assert truncated[1].role == "system"
    assert "Resumen de conversación" in truncated[1].content
    assert truncated[-1].content == "Msg 19"


def test_context_manager_create_summary_prompt():
    """Verifica creación de prompt para resumen."""
    cm = ContextManager()

    messages = [
        Message(role="system", content="Eres un asistente"),
        Message(role="user", content="Hola"),
        Message(role="assistant", content="¡Hola!"),
    ]

    prompt = cm.create_summary_prompt(messages)
    assert "Eres un asistente" in prompt
    assert "Hola" in prompt
    assert "¡Hola!" in prompt
    assert "Resume la siguiente conversación" in prompt


def test_conversation_metadata(tmp_path):
    """Verifica metadata de sesión."""
    db_path = tmp_path / "test.db"
    store = ConversationStore(db_path)

    session = store.create_session(
        name="Test", metadata={"project": "mi-agente", "tags": ["test", "dev"]}
    )

    loaded = store.load_session(session.id)
    assert loaded.metadata["project"] == "mi-agente"
    assert "test" in loaded.metadata["tags"]

    # Actualizar metadata
    store.update_session_metadata(session.id, {"project": "updated", "version": 2})
    loaded = store.load_session(session.id)
    assert loaded.metadata["project"] == "updated"
    assert loaded.metadata["version"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
