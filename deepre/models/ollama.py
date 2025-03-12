raise NotImplementedError("Ollama is not implemented yet")

"""
Ollama model provider for pydantic-ai.

This module provides a model implementation for using Ollama with pydantic-ai.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx
from pydantic_ai.providers import Provider

from deepre.models.providers import OllamaProvider

# Import pydantic-ai modules
try:
    from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        ModelResponsePart,
    )
    from pydantic_ai.models.openai import OpenAIModelSettings
except ImportError as _import_error:
    raise ImportError(
        "Please install `pydantic-ai` to use the Ollama model provider. "
        "You can install it with `pip install pydantic-ai`."
    ) from _import_error

# Import Ollama-specific types and clients
try:
    from deepre.models.providers import OllamaAsyncClient
    from ollama import ChatResponse
except ImportError as _import_error:
    raise ImportError(
        "Please install `ollama` to use the Ollama model provider. "
        "You can install it with `pip install ollama`."
    ) from _import_error


class OllamaModelSettings(OpenAIModelSettings):
    """Settings used for an Ollama model request."""

    pass


@dataclass(init=False)
class OllamaModel(Model):
    """A model that uses the Ollama API.

    This class serves as a bridge between pydantic-ai and Ollama, allowing
    you to use locally hosted Ollama models with the pydantic-ai framework.
    """

    client: OllamaAsyncClient = field(repr=False)
    _model_name: str = field(repr=False)
    _base_url: str | None = field(default=None, repr=False)

    def __init__(
        self,
        model_name: str,
        *,
        provider: OllamaProvider | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        ollama_client: OllamaAsyncClient | None = None,
        http_client: httpx.AsyncClient | None = None,
        system_prompt_role: str | None = None,
        system: str | None = "ollama",
    ):
        self._model_name = model_name
        self.client = OllamaAsyncClient(host=base_url)

        self.system_prompt_role = system_prompt_role
        self._system = system

    @property
    def model_name(self) -> str:
        """Get the name of the model.

        Returns:
            The name of the Ollama model.
        """
        return self._model_name

    @property
    def base_url(self) -> str:
        return str(self.client._client.base_url)

    @property
    def system(self) -> str | None:
        """Get the system prompt.

        Ollama doesn't have a default system prompt, so this returns None.

        Returns:
            None - Ollama doesn't have a built-in default system prompt.
        """
        return None

    async def request(
        self,
        request: ModelRequest,
        *,
        model_settings: OpenAIModelSettings | None = None,
        parameters: ModelRequestParameters | None = None,
    ) -> ModelResponse:
        """Execute a model request without streaming.

        Args:
            request: The request to execute.
            model_settings: Optional settings for the model.
            parameters: Additional parameters for the request.

        Returns:
            The model's response.

        Raises:
            NotImplementedError: This method is not implemented yet.
        """
        raise NotImplementedError("Not implemented yet")

    async def stream(
        self,
        request: ModelRequest,
        *,
        model_settings: OpenAIModelSettings | None = None,
        parameters: ModelRequestParameters | None = None,
    ) -> StreamedResponse:
        """Execute a model request with streaming.

        Args:
            request: The request to execute.
            model_settings: Optional settings for the model.
            parameters: Additional parameters for the request.

        Returns:
            A stream of the model's response parts.

        Raises:
            NotImplementedError: This method is not implemented yet.
        """
        raise NotImplementedError("Not implemented yet")


@dataclass
class OllamaStreamedResponse(StreamedResponse):
    """Response from a streaming Ollama model request.

    This class handles streaming responses from the Ollama API.
    """

    _response_stream: AsyncIterator[ChatResponse]
    _model_name: str

    @property
    def model_name(self) -> str:
        """Get the model name.

        Returns:
            The name of the model.
        """
        return self._model_name

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponsePart]:
        """Get an iterator for streaming events.

        This method is called by the StreamedResponse base class to get
        an iterator for the streaming events.

        Returns:
            An async iterator of response parts.

        Raises:
            NotImplementedError: This method is not implemented yet.
        """
        raise NotImplementedError("Not implemented yet")
