import os
from typing import Any

from openai import OpenAI

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


class DeepSeekClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.6,
        max_tokens: int = 65536,
        response_format: dict | None = None,
    ) -> str:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.6,
        max_tokens: int = 65536,
    ) -> str:
        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

    def chat_json_safe(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.6,
        max_tokens: int = 65536,
    ) -> str:
        """Try JSON mode first; fall back to plain chat on API error."""
        try:
            return self.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception:
            return self.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    def chat_multimodal(
        self,
        system_prompt: str,
        text: str,
        image_base64: str | None = None,
        temperature: float = 0.6,
        max_tokens: int = 65536,
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if image_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                }
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        return self.chat(messages, temperature, max_tokens)
