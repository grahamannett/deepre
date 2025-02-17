from functools import wraps
from typing import Any, Callable, TypeAlias

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.result import RunResult

from deepre.utils import ModelNameMixin
from deepre.app_config import GeminiConfig, OllamaConfig, conf

ModelType: TypeAlias = Model | KnownModelName
providers: dict[str, Callable[..., Model]] = {}
_providers_meta = {}


def register_provider(provider_name: str | None = None) -> Callable[[Callable], Callable]:
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        nonlocal provider_name
        if provider_name is None:
            provider_name = func.__name__.split("setup_", 1)[-1]
        providers[provider_name] = wrapper
        # _providers_meta[provider_name] = {"require_singleton": False}
        return wrapper

    return decorator


@register_provider(GeminiConfig.provider)
def setup_gemini(
    model_name: str = GeminiConfig.model_name,
    **kwargs,
) -> Model:
    return GeminiModel(model_name=model_name, **kwargs)


@register_provider(OllamaConfig.provider)
def setup_ollama(
    model_name: str = OllamaConfig.model_name,
    base_url: str = OllamaConfig.base_url,
    api_key: str = OllamaConfig.api_key,
    **kwargs,
) -> Model:
    OllamaModel = type("OllamaModel", (ModelNameMixin, OpenAIModel), {})
    return OllamaModel(model_name=model_name, api_key=api_key, base_url=base_url, **kwargs)


class LLMProvider:
    def __init__(
        self,
        provider: str = conf.ai.provider,
        model_kwargs: dict[str, Any] = {},
        agent_kwargs: dict[str, Any] = {},
    ) -> None:
        self.provider = provider

        self.agent = None
        self.model = None

        self.agent_kwargs = agent_kwargs
        self.model_kwargs = model_kwargs

    def __class_getitem__(cls, name: str) -> Callable[..., Model]:
        provider_func = providers[name]
        return provider_func

    @staticmethod
    def get_model_from_provider(provider: str, **kwargs) -> Model:
        provider_func = providers[provider]
        return provider_func(**kwargs)

    def get_model(self, **kwargs) -> Model:
        if kwargs.get("model", None) in KnownModelName:
            raise ValueError("not using `KnownModelName`: {}")

        self.model = self.get_model_from_provider(self.provider, **kwargs)
        return self.model

    def get_agent(self, model: Model | None = None, **kwargs) -> Agent:
        if (model is None) and not hasattr(self, "model"):
            model = self.get_model(**self.model_kwargs)

        if not hasattr(self, "model"):
            self.model = self.get_model()
        self.agent = Agent(self.model, **kwargs)
        return self.agent

    def run_sync(self, prompt: str, **kwargs) -> RunResult:
        """sync run of the model"""
        return self.agent.run_sync(prompt, **kwargs)

    def clear_history(self) -> None:
        """Clear the message history."""
        self.message_history = []
