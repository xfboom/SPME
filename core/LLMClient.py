# core/LLMClient.py
from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
except ModuleNotFoundError:
    ChatOpenAI = None
    HumanMessage = None

from config.settings import get_llm_api_config, get_llm_role_config


class LLMClient:
    """
    LLM client wrapper.

    Supports:
    - LangChain ChatOpenAI when langchain-openai is installed.
    - OpenAI-compatible local servers via stdlib HTTP, e.g. vLLM, LM Studio,
      llama.cpp server, or Ollama's OpenAI-compatible endpoint.
    - Native Ollama /api/chat.
    """

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        provider: str | None = None,
        role: str = "target",
    ):
        self.config = get_llm_api_config(provider, role=role, model_override=model_name)
        self.role = role
        role_config = get_llm_role_config(role)
        self.provider = self.config.provider
        self.temperature = temperature if temperature is not None else role_config.temperature if role_config.temperature is not None else 0.7
        self.model_name = self.config.model
        self.base_url = (self.config.base_url or "").rstrip("/")
        self.api_key = self.config.api_key
        self.model = None

        should_use_direct_http = self.provider in {
            "local",
            "openai_compatible",
            "vllm",
            "lmstudio",
            "ollama",
            "volc",
        }

        if not should_use_direct_http and ChatOpenAI is not None and HumanMessage is not None:
            if not self.api_key:
                raise ValueError("Missing API key. Set VOLC_API_KEY or OPENAI_API_KEY.")
            self.model = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=self.api_key,
                base_url=self.base_url,
            )

    def invoke(self, message: str) -> str:
        """
        Send a plain text message to the LLM and return its response.
        """
        if self.model is not None:
            response = self.model.invoke([HumanMessage(content=message)])
            return response.content

        if self.provider == "ollama":
            return self._invoke_ollama(message)

        return self._invoke_openai_compatible(message)

    async def ainvoke(self, message: str) -> str:
        """
        Async wrapper used by the evaluator.
        """
        if self.model is not None:
            response = await self.model.ainvoke([HumanMessage(content=message)])
            return response.content
        return await asyncio.to_thread(self.invoke, message)

    def _invoke_openai_compatible(self, message: str) -> str:
        url = self._chat_completions_url()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": message}],
            "temperature": self.temperature,
        }
        data = self._post_json(url, headers, payload)
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"LLM response has no choices: {data}")
        first = choices[0]
        if "message" in first:
            return first["message"].get("content", "")
        return first.get("text", "")

    def _invoke_ollama(self, message: str) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": message}],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        data = self._post_json(url, {"Content-Type": "application/json"}, payload)
        if "message" in data:
            return data["message"].get("content", "")
        return data.get("response", "")

    def _chat_completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        if self.base_url.endswith(("/v1", "/v3")):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    @staticmethod
    def _post_json(url: str, headers: dict, payload: dict) -> dict:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP error {exc.code} from {url}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach LLM server at {url}: {exc}") from exc
        return json.loads(body)
