from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic_ai.messages import ModelMessage

from deepre.provider import LLMProvider, register_provider


async def async_generator():
    for chunk in ["Test ", "stream ", "response"]:
        yield chunk


@pytest.fixture
async def mock_model():
    model = Mock()
    model.generate = AsyncMock(return_value="Test response")
    model.stream = AsyncMock()
    chunks = [chunk async for chunk in async_generator()]
    model.stream.return_value = chunks
    return model


@pytest.fixture
async def provider(mock_model):
    with patch("deepre.researcher.provider.providers") as mock_providers:
        mock_providers["ollama"] = Mock(return_value=mock_model)
        provider = LLMProvider()
        return provider


@pytest.mark.asyncio
async def test_generate_response(provider):
    response = await provider.generate("Test prompt")
    assert response == "Test response"
    assert len(provider.message_history) == 2
    assert provider.message_history[0].role == "user"
    assert provider.message_history[0].content == "Test prompt"
    assert provider.message_history[1].role == "assistant"
    assert provider.message_history[1].content == "Test response"


@pytest.mark.asyncio
async def test_stream_response(provider):
    chunks = []
    async for chunk in provider.generate("Test prompt", stream=True):
        chunks.append(chunk)

    assert chunks == ["Test ", "stream ", "response"]
    assert len(provider.message_history) == 1
    assert provider.message_history[0].role == "user"
    assert provider.message_history[0].content == "Test prompt"


def test_clear_history(provider):
    provider.message_history = [
        ModelMessage(role="user", content="Test prompt"),
        ModelMessage(role="assistant", content="Test response"),
    ]
    provider.clear_history()
    assert len(provider.message_history) == 0


def test_register_provider():
    test_providers = {}
    with patch("deepre.researcher.provider.providers", test_providers):

        @register_provider("test")
        def setup_test():
            return "test_model"

        assert "test" in test_providers
        assert test_providers["test"]() == "test_model"


@pytest.mark.parametrize(
    "provider_name,prompt",
    [("ollama", "What is the capital of France?")],
)
def test_provider(provider_name, prompt):
    provider = LLMProvider(provider=provider_name)
    resp = provider.run_sync(prompt)
    assert resp.usage().requests == 1
