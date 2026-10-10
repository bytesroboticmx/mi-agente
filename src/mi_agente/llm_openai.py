"""Cliente OpenAI para el agente."""

import json
from typing import Any, Iterator

from openai import OpenAI

from mi_agente.config import Settings
from mi_agente.llm import LLMClient, LLMResponse, Message, ToolSpec


class OpenAIClient(LLMClient):
    """Cliente para API de OpenAI (y compatibles)."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.client = OpenAI(
            api_key=settings.get_api_key() or None,
            base_url=settings.base_url,
            timeout=60.0,
        )

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse | Iterator[LLMResponse]:
        openai_messages = [m.to_openai() for m in messages]
        openai_tools = [t.to_openai() for t in tools] if tools else None

        if stream:
            return self._stream_complete(openai_messages, openai_tools, **kwargs)

        response = self.client.chat.completions.create(
            model=self.settings.model_name,
            messages=openai_messages,
            tools=openai_tools,
            tool_choice="auto" if openai_tools else None,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            **kwargs,
        )

        choice = response.choices[0]
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]

        return LLMResponse(
            content=choice.message.content,
            tool_calls=tool_calls,
            usage=response.usage.model_dump() if response.usage else None,
            model=response.model,
            finish_reason=choice.finish_reason,
        )

    def _stream_complete(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        **kwargs,
    ) -> Iterator[LLMResponse]:
        stream = self.client.chat.completions.create(
            model=self.settings.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else None,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            stream=True,
            **kwargs,
        )

        accumulated_content = ""
        accumulated_tool_calls: list[dict] = []

        for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                accumulated_content += delta.content
                yield LLMResponse(content=delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx >= len(accumulated_tool_calls):
                        accumulated_tool_calls.append({
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        })
                    if tc.function.name:
                        accumulated_tool_calls[idx]["function"]["name"] = tc.function.name
                    if tc.function.arguments:
                        accumulated_tool_calls[idx]["function"]["arguments"] += tc.function.arguments

            if choice.finish_reason:
                yield LLMResponse(
                    content=accumulated_content if accumulated_content else None,
                    tool_calls=accumulated_tool_calls if accumulated_tool_calls else None,
                    finish_reason=choice.finish_reason,
                    model=chunk.model,
                )

    async def acomplete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse | Any:
        # TODO: Implementar async con openai.AsyncOpenAI
        return self.complete(messages, tools, stream, **kwargs)

    def count_tokens(self, messages: list[Message]) -> int:
        # Aproximación simple: ~4 chars por token
        total_chars = sum(len(m.content or "") for m in messages)
        for m in messages:
            if m.tool_calls:
                total_chars += sum(len(str(tc)) for tc in m.tool_calls)
        return total_chars // 4