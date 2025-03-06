from dataclasses import dataclass, field
from os import environ
from pathlib import Path
from typing import Any

OLLAMA_URL = environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_TIMEOUT = float(environ.get("DEFAULT_TIMEOUT", "30.0"))


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
    """Base configuration for external services that only require an API key."""

    api_key: str


@dataclass
class EndpointExternalServiceConfig(ExternalServiceConfig):
    """Configuration for external services that require both API key and base URL."""

    base_url: str


GeminiConfig = AIConfig(
    provider="gemini",
    model_name="gemini-1.5-flash",
)

OllamaConfig = AIConfig(
    provider="ollama",
    model_name="llama3.2:3b",
    api_key="ollama-api-key",
    base_url=f"{OLLAMA_URL}/v1",
)


SerpConfig = EndpointExternalServiceConfig(
    api_key=environ.get("SERPAPI_API_KEY", ""),
    base_url="https://serpapi.com/search",
)

JinaConfig = EndpointExternalServiceConfig(
    api_key=environ.get("JINA_API_KEY", ""),
    base_url="https://r.jina.ai/",
)

LogFireConfig = ExternalServiceConfig(
    api_key=environ.get("LOGFIRE_API_KEY", ""),
)


class AppConfig:
    ai = OllamaConfig
    serpapi = SerpConfig
    jina = JinaConfig
    logfire = LogFireConfig

    # allows for smaller templates/models to be used during dev these are still 3B+
    # models since it seems like you want models that are capable of producing coherent
    # long-form answers to get reliable results. alt `from reflex.utils.exec import is_prod_mode`
    smol: bool = True

    tomls_dir = Path(__file__).parent / "tomls"
    model_configs_file = f"{'s.' if smol else ''}model_configs.toml"


conf = AppConfig()
