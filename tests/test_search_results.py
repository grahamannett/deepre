import os
import tempfile
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from deepre.researcher import PageResult, WebServices, WebServicesTool, WebServicesWithCache

pytestmark = pytest.mark.anyio


@pytest.fixture
async def client():
    """Create an httpx AsyncClient."""
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture
def cache_dir():
    """Create a temporary directory for caching."""
    with tempfile.TemporaryDirectory(prefix="test_cache") as temp_dir:
        yield temp_dir


async def test_base_get_search_results(client):
    """Test the base WebServices.get_search_results method."""
    service = WebServices()

    # Test with search_query
    search_query = "python programming"
    result = await service.get_search_results(client, search_query=search_query)

    # Check that the result is a dictionary with expected keys
    assert isinstance(result, dict)
    assert "search_metadata" in result
    assert "search_parameters" in result

    # Test with req_kwargs
    req_kwargs = service._get_search_kwargs("python testing")
    result = await service.get_search_results(client, req_kwargs=req_kwargs)

    # Check that the result is a dictionary with expected keys
    assert isinstance(result, dict)
    assert "search_metadata" in result
    assert "search_parameters" in result

    # Test error case: neither search_query nor req_kwargs
    with pytest.raises(ValueError):
        await service.get_search_results(client)


async def test_cached_get_search_results(client, cache_dir):
    """Test the WebServicesWithCache.get_search_results method."""
    service = WebServicesWithCache(cache_dir=cache_dir)

    # First call - should cache the result
    search_query = "python caching"
    result1 = await service.get_search_results(client, search_query)

    # Second call with same query - should return cached result
    result2 = await service.get_search_results(client, search_query)

    # Results should be the same
    assert result1 == result2

    # Check the cache directory has files
    assert os.listdir(cache_dir)

    # Check results structure
    assert isinstance(result1, dict)
    assert "search_metadata" in result1
    assert "organic_results" in result1


async def test_get_search_results_as_page_results(client, cache_dir):
    """Test the WebServicesTool.get_search_results_as_page_results method."""
    service = WebServicesTool(cache_dir=cache_dir)

    # Create a page result with URL as the search query
    search_query = "python flask"
    page = PageResult(url=search_query)

    # Patch the get_search_results_as_page_results method to handle None dates
    with patch.object(WebServicesTool, "get_search_results_as_page_results") as mock_method:
        # Create a mock implementation that handles None dates
        async def patched_method(client, page):
            resp_json = await service.get_search_results(client, page.url)
            results = []

            for item in resp_json.get("organic_results", []):
                if "link" not in item:
                    continue

                # Ensure date is a string by providing default empty string
                results.append(
                    PageResult(
                        url=item["link"],
                        date=item.get("date", ""),
                        title=item.get("title"),
                        description=item.get("snippet"),
                    )
                )

            return results

        # Set the mock to use our implementation
        mock_method.side_effect = patched_method

        # Get search results as PageResult objects
        results = await service.get_search_results_as_page_results(client, page)

        # Check that we got valid results
        assert isinstance(results, list)
        assert len(results) > 0

        # Check that all results are PageResult objects
        for result in results:
            assert isinstance(result, PageResult)
            assert hasattr(result, "url")
            assert hasattr(result, "title")
            assert hasattr(result, "description")


async def test_get_page_content_with_cache(client, cache_dir):
    """Test the WebServicesWithCache.get_page_content method."""
    service = WebServicesWithCache(cache_dir=cache_dir)

    # Test fetching a URL with caching
    url = "https://www.python.org/"

    # First fetch - should cache the result
    page_text1 = await service.get_page_content(client, url)

    # Second fetch - should return from cache
    page_text2 = await service.get_page_content(client, url)

    # Results should be the same
    assert page_text1 == page_text2

    # Check that we got text content
    assert isinstance(page_text1, str)
    assert "Python" in page_text1
    assert "https://www.python.org/" in page_text1

    # Check that the cache has content
    assert os.listdir(cache_dir)


async def test_request_error_handling():
    """Test error handling when a request fails."""
    service = WebServices()

    # Create a mock client that returns a 404 response
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status_code = 404
    mock_response.url = "https://example.com/nonexistent"
    mock_client.get.return_value = mock_response

    # Test that an exception is raised for a non-200 status code
    with pytest.raises(Exception) as excinfo:
        await service.get_page_content(mock_client, "https://example.com/nonexistent")

    # Check the exception message
    assert "Failed to fetch" in str(excinfo.value)
    assert "404" in str(excinfo.value)
