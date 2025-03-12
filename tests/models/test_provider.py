import httpx


import ollama

from deepre.models.providers import OllamaProvider


# pytestmark = pytest.mark.skipif(not imports_successful(), reason="openai not installed")


def test_ollama_provider():
    provider = OllamaProvider()
    assert provider.name == "ollama"
    assert provider.base_url == "http://127.0.0.1:11434"
    assert isinstance(provider.client, ollama.AsyncClient)


def test_ollama_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = OllamaProvider(http_client=http_client)
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_ollama_cached_http_client() -> None:
    # this should fail until _patch_client gives the `cached_async_http_client`
    # but that needs to be done to allow the base_url to be correct.
    ollama_client = ollama.AsyncClient()
    provider = OllamaProvider()
    assert provider.client == ollama_client


if __name__ == "__main__":
    test_ollama_provider()
