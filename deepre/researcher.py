import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

import httpx
from pydantic_ai import Tool

from deepre.app_config import conf
from deepre.utils.cache_utils import get_from_cache, hash_str, save_to_cache

# Set up logger
logger = logging.getLogger(__name__)


def default_resp_cb(resp: httpx.Response) -> dict:
    return resp.json()


def jina_resp_cb(resp: httpx.Response) -> str:
    """
    this is the equivalent of not using `"Accept": "application/json"` in the headers
    """
    data = resp.json()["data"]
    title, url, published_time, content = data["title"], data["url"], data["publishedTime"], data["content"]
    return f"Title: {title}\n\nURL Source: {url}\n\nPublished Time: {published_time}\n\nMarkdown Content:\n{content}"


def _check_resp(resp: httpx.Response) -> httpx.Response:
    """Check if a response is valid.

    Args:
        resp: Response to check

    Returns:
        The response if it is valid (status code 200), raises an exception otherwise
    """
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch {resp.url}: {resp.status_code}")

    return resp


def deep_merge(base: dict, override: dict | None = None) -> dict:
    """Recursively merge dictionaries.

    The override dictionary values take precedence over base values,
    except for nested dictionaries which are merged recursively.

    Args:
        base: Base dictionary
        override: Dictionary with values to override

    Returns:
        Merged dictionary
    """
    if not override:
        return base

    result = base.copy()

    for key, value in override.items():
        # If both are dicts, merge them recursively
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:  # Otherwise, override the value
            result[key] = value

    return result


class HTTPToolMeta:
    name: str
    description: str
    # allow client to be set after init, makes it easier to use with pydantic-ai or for threading
    client: httpx.AsyncClient | None = None

    _post_get_cb: Callable = staticmethod(default_resp_cb)  # Changed to staticmethod to avoid passing self

    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def get_kwargs(self, *args, **kwargs) -> dict[str, Any]:
        """Generate base kwargs for the HTTP request. This method should be implemented
        by subclasses to return the base kwargs specific for the service.
        """
        raise NotImplementedError

    async def _execute_request(self, *args, **kwargs) -> dict:
        """Execute an HTTP request and process the response.

        This is a shared implementation that handles the common HTTP request lifecycle:
        1. Prepares request kwargs using _make_req_kwargs
        2. Gets a response with get_response
        3. Processes the response with _post_get_cb

        Args:
            *args: Positional arguments for get_kwargs
            **kwargs: Keyword arguments for get_kwargs

        Returns:
            The processed response from _post_get_cb
        """
        req_kwargs = self._make_req_kwargs(*args, **kwargs)
        resp = await self._get_response(**req_kwargs)
        return self._post_get_cb(resp)

    async def _get_response(self, **req_kwargs) -> httpx.Response:
        """Get an HTTP response from the API.

        Args:
            **req_kwargs: Keyword arguments to pass to get_kwargs

        Returns:
            HTTP response
        """
        if not self.client:
            raise ValueError("Must set client on init or after")

        resp = await self.client.get(**req_kwargs)
        return _check_resp(resp)

    def _make_req_kwargs(self, *args, merge_kwargs: dict | None = None, **kwargs) -> dict[str, Any]:
        """
        This method retrieves the base HTTP request arguments and merges them with any additional
        keyword arguments provided. The merged kwargs can be used later to check the cache for
        specific request configurations, ensuring that the correct parameters are used for the
        HTTP request.
        """
        base_kwargs = self.get_kwargs(*args, **kwargs)
        return deep_merge(base_kwargs, merge_kwargs)

    def as_tool(self, **kwargs) -> Tool:
        tool_kwargs = {"takes_ctx": False, "name": self.name, "description": self.description} | kwargs
        return Tool(function=self.__call__, **tool_kwargs)


@dataclass
class SearchEngineTool(HTTPToolMeta):
    """Tool for performing web searches.

    This tool executes web searches using search engines like Google via SerpAPI.
    It can be used as a standalone function or integrated with an LLM agent.
    """

    service = conf.serpapi

    name: str = "search_engine"
    description: str = "Search the web for information on a given topic"

    async def __call__(self, search_query: str, merge_kwargs: dict | None = None) -> dict:
        return await self._execute_request(search_query=search_query, merge_kwargs=merge_kwargs)

    def get_kwargs(self, search_query: str) -> dict[str, Any]:
        """Create base kwargs for search engine request."""
        return {
            "url": self.service.base_url,
            "params": {"q": search_query, "engine": "google", "api_key": self.service.api_key},
        }


@dataclass
class FetchPageTool(HTTPToolMeta):
    """Tool for fetching a web page's content."""

    service = conf.jina
    name: str = "fetch_page"
    description: str = "Fetch the full text content of a webpage by URL"

    async def __call__(self, url: str, merge_kwargs: dict | None = None) -> dict:
        return await self._execute_request(url=url, merge_kwargs=merge_kwargs)

    def get_kwargs(self, url: str) -> dict[str, Any]:
        """Create base kwargs for page fetch request. Using `Accept: application/json`
        to get the json response, without this it will return a text response that is
        equivalent to `jina_resp_cb`. Returning json makes it easier for later services/caching
        """
        return {
            "url": f"{self.service.base_url}/{url}",
            "headers": {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.service.api_key}",
            },
        }


class CachedHTTPMixin(HTTPToolMeta):
    """Mixin that adds caching to HTTPToolMeta subclasses.  This ISNT a subclass of HTTPToolMeta,
    but defining the type correctly is a lot of bloat
    Note: This mixin should only be used with HTTPToolMeta subclasses.
    """

    _check_cache: bool = True
    _save_cache: bool = True
    _mark_from_cache: bool = True

    async def _execute_with_cache(self, **kwargs) -> dict:
        """Execute a request with caching.
        TODO: allow req_kwargs to be passed into __call__

        This method relies on methods from HTTPToolMeta subclasses.
        """
        req_kwargs = self._make_req_kwargs(**kwargs)
        req_hash = hash_str(str(req_kwargs))

        if res := get_from_cache(req_hash, callback=json.loads, check_cache=self._check_cache):
            if self._mark_from_cache:
                res["from_cache"] = True
            return res
        res = await super().__call__(**kwargs)

        save_to_cache(req_hash, res, callback=json.dumps, save_cache=self._save_cache)
        return res


@dataclass
class CachedSearchEngineTool(CachedHTTPMixin, SearchEngineTool):
    """Add merge_kwargs to the call and execute after checking that pydantic-ai works, e.g.
    async def __call__(self, search_query: str, merge_kwargs: dict | None = None) -> dict:
        return await self._execute_with_cache(search_query=search_query, merge_kwargs=merge_kwargs)
    """

    async def __call__(self, search_query: str) -> dict:
        return await self._execute_with_cache(search_query=search_query)


@dataclass
class CachedFetchPageTool(CachedHTTPMixin, FetchPageTool):
    async def __call__(self, url: str) -> dict:
        return await self._execute_with_cache(url=url)


class Researcher:
    pass
