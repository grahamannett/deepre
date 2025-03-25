from __future__ import annotations as _annotations

from typing import TypeVar
import httpx
from importlib import metadata
from functools import wraps


try:
    from ollama import AsyncClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError("Please install `ollama` to use the Ollama provider") from _import_error


try:
    __version__ = metadata.version("ollama")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"
from pydantic_ai.providers import Provider

InterfaceClient = TypeVar("InterfaceClient")


def _patch_client(fn, _use_cached_client: bool = True):
    @wraps(fn)
    def wrapper(**kwargs):
        if "http_client" in kwargs:
            return kwargs["http_client"]
        return fn(**kwargs)

    return wrapper


class OllamaAsyncClient(AsyncClient):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        _patched_client = _patch_client(httpx.AsyncClient)
        super(AsyncClient, self).__init__(client=_patched_client, **kwargs)


class OllamaProvider(Provider[OllamaAsyncClient]):
    """
    Provider for Ollama API.

    Todo: Find a way to use `cached_async_http_client`
    """

    @property
    def name(self) -> str:
        return "ollama"  # pragma: no cover

    @property
    def base_url(self) -> str:
        return str(self.client._client.base_url)

    @property
    def client(self) -> OllamaAsyncClient:
        return self._client

    def __init__(self, **kwargs) -> None:
        """Create a new Ollama provider."""
        self._client = OllamaAsyncClient(**kwargs)
