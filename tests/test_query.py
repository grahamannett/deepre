# import aiohttp
import httpx
import pytest
from _stubs import (
    delegated_model_output_extracted_context,
    model_output_extracted_context,
    serp_link,
    serp_queries,
    serp_query,
    serp_resp,
    user_query,
    webpage_text,
)

from deepre import query
from deepre.provider import LLMProvider

# Configure pytest to handle async tests
pytestmark = pytest.mark.anyio


@pytest.fixture
async def client_session():
    """Create an aiohttp client session."""
    async with httpx.AsyncClient() as client:
        yield client


# Model configurations
@pytest.fixture
def model_configs():
    return {
        "reasoning": {
            # "model_name": "deepseek-r1:70b",
            "model_name": "deepseek-r1:32b",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama-api-key",
        },
        "tool": {
            "model_name": "llama3.3:70b",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama-api-key",
        },
    }


@pytest.fixture
def reasoning_model(model_configs):
    # return LLMProvider.get_model_from_provider("ollama", **model_configs["reasoning"])
    return LLMProvider["ollama"](**model_configs["reasoning"])


@pytest.fixture
def tool_model(model_configs):
    # return LLMProvider.get_model_from_provider("ollama", **model_configs["tool"])
    return LLMProvider["ollama"](**model_configs["tool"])


@pytest.fixture
async def mock_fetch_webpage_text(monkeypatch):
    """Mock the fetch_webpage_text function to return stub webpage text."""

    async def mock_fetch(*args, **kwargs):
        return webpage_text

    monkeypatch.setattr(query, "fetch_webpage_text", mock_fetch)


@pytest.fixture
async def mock_perform_search(monkeypatch):
    async def mock_perform_search(*args, **kwargs):
        return serp_resp

    monkeypatch.setattr(query, "perform_search", mock_perform_search)


@pytest.fixture
async def mock_extract_queries_from_text(monkeypatch):
    async def mock_extract_queries_from_text(*args, **kwargs):
        return serp_queries

    monkeypatch.setattr(query, "extract_queries_from_text", mock_extract_queries_from_text)


@pytest.fixture
async def mock_generate_search_queries(monkeypatch):
    async def mock_generate_search_queries(*args, **kwargs):
        return serp_queries

    monkeypatch.setattr(query, "generate_search_queries", mock_generate_search_queries)


@pytest.fixture
async def mock_is_page_useful(monkeypatch):
    async def mock_is_page_useful(*args, **kwargs):
        return True

    monkeypatch.setattr(query, "is_page_useful", mock_is_page_useful)


@pytest.fixture
async def mock_extract_relevant_context(monkeypatch):
    async def mock_extract_relevant_context(*args, **kwargs):
        return model_output_extracted_context

    monkeypatch.setattr(query, "extract_relevant_context", mock_extract_relevant_context)


async def test_generate_search_queries(reasoning_model, tool_model):
    model_resp = await query.generate_search_queries(model=reasoning_model, user_query=user_query)
    queries = await query.extract_queries_from_text(model=tool_model, user_query=model_resp)


async def test_perform_search(mock_perform_search, client_session) -> None:
    # async with aiohttp.ClientSession() as session:
    results = await query.perform_search(client_session, query=serp_query)
    assert len(results) > 1


async def test_fetch_webpage_text(mock_fetch_webpage_text, client_session) -> None:
    text = await query.fetch_webpage_text(client_session, url=serp_link)
    assert text == webpage_text


async def test_process_link(
    mock_fetch_webpage_text,
    mock_is_page_useful,
    mock_extract_relevant_context,
    client_session,
    tool_model,
    reasoning_model,
) -> None:
    """Test process_link with mocked webpage fetch."""
    result = await query.process_link(
        client=client_session,
        link=serp_link,
        search_query=serp_query,
        user_query=user_query,
        tool_model=tool_model,
        reasoning_model=reasoning_model,
        force_is_useful=True,  # Force the page to be considered useful for testing
    )
    assert result is not None


async def test_is_page_useful(tool_model):
    resp = await query.is_page_useful(
        model=tool_model,
        user_query=user_query,
        webpage_text=webpage_text,
    )


async def test_extract_relevant_context(reasoning_model):
    resp = await query.extract_relevant_context(
        model=reasoning_model,
        user_query=user_query,
        search_query=serp_query,
        page_text=webpage_text,
    )


async def test_delegate_extract_relevant_context(tool_model):
    resp = await query.delegate_extract_relevant_context(
        model=tool_model,
        model_output=model_output_extracted_context,
    )


async def test_extract_and_delegate_relevant_context(reasoning_model, tool_model):
    model_output = await query.extract_relevant_context(
        model=reasoning_model,
        user_query=user_query,
        search_query=serp_query,
        page_text=webpage_text,
    )
    resp = await query.delegate_extract_relevant_context(
        model=tool_model,
        model_output=model_output,
    )


async def test_get_new_search_queries(tool_model):
    new_queries = await query.get_new_search_queries(
        model=tool_model,
        # model=reasoning_model,
        user_query=user_query,
        previous_queries=serp_queries,
        contexts=[delegated_model_output_extracted_context, delegated_model_output_extracted_context],
    )


async def test_generate_final_report(reasoning_model):
    resp = await query.generate_final_report(
        model=reasoning_model,
        user_query=user_query,
        contexts=[delegated_model_output_extracted_context, delegated_model_output_extracted_context],
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_generate_search_queries())
    asyncio.run(test_perform_search())
    asyncio.run(test_fetch_webpage_text())
    asyncio.run(test_is_page_useful())
    asyncio.run(test_extract_relevant_context())
    asyncio.run(test_delegate_extract_relevant_context())
    asyncio.run(test_extract_and_delegate_relevant_context())
    asyncio.run(test_get_new_search_queries())
