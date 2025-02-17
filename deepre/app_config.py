from os import environ
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AIConfig:
    provider: str
    model_name: str
    api_key: str = ""
    base_url: str = ""
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    agent_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExternalServiceConfig:
    api_key: str
    base_url: str


GeminiConfig = AIConfig(
    provider="gemini",
    model_name="gemini-1.5-flash",
)

OllamaConfig = AIConfig(
    provider="ollama",
    model_name="llama3.2:3b",
    api_key="ollama-api-key",
    base_url="http://localhost:11434/v1",
)


SerpConfig = ExternalServiceConfig(
    api_key=environ.get("SERPAPI_API_KEY", ""),
    base_url="https://serpapi.com/search",
)

JinaConfig = ExternalServiceConfig(
    api_key=environ.get("JINA_API_KEY", ""),
    base_url="https://r.jina.ai/",
)


class AppConfig:
    ai = OllamaConfig
    serpapi = SerpConfig
    jina = JinaConfig


conf = AppConfig()
