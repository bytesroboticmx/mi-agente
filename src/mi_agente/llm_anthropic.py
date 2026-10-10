"""Cliente Anthropic para el agente."""

from typing import Any, Iterator

from anthropic import Anthropic

from mi_agente.config import Settings
from mi_agente.llm import LLMClient, LLMResponse, Message, ToolSpec


class AnthropicClient(LLMClient):
    """Cliente para API de Anthropic (Claude)."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.client = Anthropic(
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
        # Separar system prompt
        system_prompt = ""
        anthropic_messages = []
        for m in messages:
            if m.role == "system":
                system_prompt = m.content or ""
            else:
                anthropic_messages.append(m.to_anthropic())

        anthropic_tools = [t.to_anthropic() for t in tools] if tools else None

        if stream:
            return self._stream_complete(system_prompt, anthropic_messages, anthropic_tools, **kwargs)

        response = self.client.messages.create(
            model=self.settings.model_name,
            system=system_prompt or None,
            messages=anthropic_messages,
            tools=anthropic_tools,
            tool_choice={"type": "auto"} if anthropic_tools else None,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            **kwargs,
        )

        content = None
        tool_calls = None

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            model=response.model,
            finish_reason=response.stop_reason,
        )

    def _stream_complete(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict] | None,
        **kwargs,
    ) -> Iterator[LLMResponse]:
        import json

        stream = self.client.messages.stream(
            model=self.settings.model_name,
            system=system_prompt or None,
            messages=messages,
            tools=tools,
            tool_choice={"type": "auto"} if tools else None,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            **kwargs,
        )

        accumulated_content = ""
        accumulated_tool_calls: dict[int, dict] = {}

        with stream as s:
            for event in s:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        accumulated_content += event.delta.text
                        yield LLMResponse(content=event.delta.text)
                    elif event.delta.type == "input_json_delta":
                        idx = event.index
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        accumulated_tool_calls[idx]["function"]["arguments"] += event.delta.partial_json
                elif event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        idx = event.index
                        accumulated_tool_calls[idx] = {
                            "id": event.content_block.id,
                            "type": "function",
                            "function": {
                                "name": event.content_block.name,
                                "arguments": "",
                            },
                        }
                elif event.type == "message_stop":
                    tool_calls = list(accumulated_tool_calls.values()) if accumulated_tool_calls else None
                    yield LLMResponse(
                        content=accumulated_content if accumulated_content else None,
                        tool_calls=tool_calls,
                        finish_reason="tool_use" if tool_calls else "end_turn",
                        model=self.settings.model_name,
                    )

    async def acomplete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        stream: bool = False,
        **kwargs,
    ) -> LLMResponse | Any:
        # TODO: Implementar async con anthropic.AsyncAnthropic
        return self.complete(messages, tools, stream, **kwargs)

    def count_tokens(self, messages: list[Message]) -> int:
        # Usar API de conteo de Anthropic si está disponible, sino aproximación
        try:
            system_prompt = ""
            anthropic_messages = []
            for m in messages:
                if m.role == "system":
                    system_prompt = m.content or ""
                else:
                    anthropic_messages.append(m.to_anthropic())

            response = self.client.messages.count_tokens(
                model=self.settings.model_name,
                system=system_prompt or None,
                messages=anthropic_messages,
            )
            return response.input_tokens
        except Exception:
            # Fallback: aproximación
            total_chars = sum(len(m.content or "") for m in messages)
            return total_chars // 4