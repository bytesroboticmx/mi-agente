"""Cliente Ollama para el agente (modelos locales)."""

import json
from typing import Any, Iterator

import httpx

from mi_agente.config import Settings
from mi_agente.llm import LLMClient, LLMResponse, Message, ToolSpec


class OllamaClient(LLMClient):
    """Cliente para Ollama (modelos locales)."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        base = settings.base_url or "http://localhost:11434"
        self.base_url = base.rstrip("/")
        self.client = httpx.Client(timeout=120.0)

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse | Iterator[LLMResponse]:
        ollama_messages = [m.to_openai() for m in messages]
        ollama_tools = [t.to_openai() for t in tools] if tools else None

        payload = {
            "model": self.settings.model_name,
            "messages": ollama_messages,
            "stream": stream,
            "options": {
                "temperature": self.settings.temperature,
                "num_predict": self.settings.max_tokens,
            },
        }
        if ollama_tools:
            payload["tools"] = ollama_tools

        if stream:
            return self._stream_complete(payload)

        response = self.client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        return self._parse_response(data)

    def _stream_complete(self, payload: dict) -> Iterator[LLMResponse]:
        with self.client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                yield self._parse_stream_chunk(data)

    def _parse_response(self, data: dict) -> LLMResponse:
        msg = data.get("message", {})
        content = msg.get("content")
        tool_calls = None

        if "tool_calls" in msg:
            tool_calls = []
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": json.dumps(func.get("arguments", {})),
                    },
                })

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage={
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"),
            },
            model=data.get("model"),
            finish_reason=data.get("done_reason"),
        )

    def _parse_stream_chunk(self, data: dict) -> LLMResponse:
        msg = data.get("message", {})
        content = msg.get("content")
        tool_calls = None

        if "tool_calls" in msg:
            tool_calls = []
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": json.dumps(func.get("arguments", {})),
                    },
                })

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=data.get("done_reason"),
            model=data.get("model"),
        )

    async def acomplete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse | Any:
        # TODO: Implementar async con httpx.AsyncClient
        return self.complete(messages, tools, stream, **kwargs)

    def count_tokens(self, messages: list[Message]) -> int:
        # Ollama no tiene API de conteo, aproximación
        total_chars = sum(len(m.content or "") for m in messages)
        for m in messages:
            if m.tool_calls:
                total_chars += sum(len(str(tc)) for tc in m.tool_calls)
        return total_chars // 4