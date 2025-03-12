import json
from typing import Any, Callable, Optional

import httpx
from pydantic_ai import Tool

from deepre.app_config import EndpointExternalServiceConfig, conf
from deepre.utils.cache_utils import get_from_cache, hash_str, save_to_cache


def default_json_resp(resp: httpx.Response) -> dict:
    """Parse JSON response from an HTTP request.

    Args:
        resp: The HTTP response to parse

    Returns:
        The parsed JSON as a dictionary
    """
    return resp.json()


def jina_resp_cb(resp: httpx.Response) -> str:
    """Format Jina API response as a string.

    This is the equivalent of not using `"Accept": "application/json"` in the headers.

    Args:
        resp: The HTTP response from Jina

    Returns:
        A formatted string with the article title, URL, published time, and content
    """
    data = resp.json()["data"]
    title, url, published_time, content = data["title"], data["url"], data["publishedTime"], data["content"]
    return f"Title: {title}\n\nURL Source: {url}\n\nPublished Time: {published_time}\n\nMarkdown Content:\n{content}"


def _check_resp(resp: httpx.Response) -> httpx.Response:
    """Check if a response is valid.

    Args:
        resp: Response to check

    Returns:
        The response if it is valid (status code 200)

    Raises:
        Exception: If the status code is not 200
    """
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch {resp.url}: {resp.status_code}")

    return resp


def deep_merge(base: dict, updates: Optional[dict] = None, **kwargs) -> dict:
    """Recursively merge dictionaries.

    The updates dictionary values and any additional keyword arguments take precedence over
    base values, except for nested dictionaries which are merged recursively.

    Args:
        base: Base dictionary
        updates: Dictionary with values to update/extend the base dict
        **kwargs: Additional key-value pairs to update/extend the base dict

    Returns:
        Merged dictionary
    """
    if (not updates) and (not kwargs):
        return base

    # Create a working copy of updates
    if (not updates) and kwargs:  # handle when kwargs used but no updates dict
        updates = kwargs
    elif updates and kwargs:  # handle when both updates and kwargs are passed in
        updates = deep_merge(updates, kwargs)

    result = base.copy()

    # At this point, updates is guaranteed to be a dictionary
    assert updates is not None

    for key, value in updates.items():
        # If both are dicts, merge them recursively
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:  # Otherwise, override the value
            result[key] = value

    return result


class ToolMixin:
    """Base class for all tools providing common functionality.

    This class provides initialization, equality, hashing, and other
    utility methods that all tools can inherit.
    """

    name: str
    description: str

    def __init__(self, **kwargs):
        """Initialize a tool with kwargs.

        Any passed kwargs will be set as attributes on the instance.

        Args:
            **kwargs: Key-value pairs to set as attributes
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Allow for subclasses to have their own post init
        self.__post_init__()

    def __post_init__(self) -> None:
        """Post-initialization hook for subclasses to customize."""
        pass

    def __repr__(self) -> str:
        """Create a string representation of the tool.

        Returns:
            A string showing the class name and attributes
        """
        return f"{self.__class__.__name__}({self.__dict__})"

    def __eq__(self, other) -> bool:
        """Compare two tools for equality by comparing their attributes.

        Args:
            other: Another tool to compare with

        Returns:
            True if the tools have identical attributes, False otherwise
        """
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        """Generate a hash value for the tool.

        Returns:
            A hash value based on the tool's attributes
        """
        return hash(tuple(sorted(self.__dict__.items())))

    async def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool with the given arguments.

        This is a helper method that delegates to run() for easier usage/debugging.

        Args:
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool

        Returns:
            The result of running the tool
        """
        return await self.run(*args, **kwargs)

    @property
    def __name__(self) -> str:
        """Get the name of the tool.

        Returns:
            The name of the tool
        """
        return self.name

    async def run(self, *args, **kwargs) -> Any:
        """Execute the tool's primary functionality.

        Must be implemented by subclasses.

        Args:
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool

        Returns:
            The result of running the tool

        Raises:
            NotImplementedError: If not implemented by a subclass
        """
        raise NotImplementedError("Must be implemented by subclasses")

    def as_tool(self, **kwargs) -> Tool:
        """Convert this tool to a pydantic_ai Tool for use with agents.

        Args:
            **kwargs: Additional arguments to pass to Tool constructor

        Returns:
            A pydantic_ai Tool wrapping this tool's run method
        """
        tool_kwargs = {"takes_ctx": False, "name": self.name, "description": self.description} | kwargs
        return Tool(function=self.run, **tool_kwargs)


class HTTPToolMeta(ToolMixin):
    """Base class for HTTP-based tools.

    This class provides common functionality for tools that make HTTP requests.
    """

    service: EndpointExternalServiceConfig
    client: httpx.AsyncClient | None = None  # HTTP client for making requests

    # Response processing callback
    _post_get_cb: Callable = staticmethod(default_json_resp)

    # Extra kwargs to merge into the http_kwargs
    _extra_kwargs: dict | None = None

    def http_kwargs(self, **kwargs) -> dict:
        """Get HTTP request kwargs.

        This method must be implemented by subclasses to provide the specific
        HTTP request parameters needed for the service.

        Args:
            **kwargs: Parameters needed to construct the HTTP request

        Returns:
            Dictionary of HTTP request parameters

        Raises:
            NotImplementedError: If not implemented by a subclass
        """
        raise NotImplementedError("Must be implemented by each service class")

    async def execute_get(self, *args, **kwargs) -> dict:
        """Execute an HTTP GET request and process the response.

        Args:
            *args: Positional arguments for http_kwargs
            **kwargs: Keyword arguments for http_kwargs or http_req_kwargs

        Returns:
            Processed response data

        Raises:
            ValueError: If client is not set
        """
        if not self.client:
            raise ValueError("Must set client on init or after")

        if "http_req_kwargs" in kwargs:
            http_req_kwargs = kwargs.pop("http_req_kwargs")
        else:
            http_req_kwargs = self.http_kwargs(*args, **kwargs)

        resp = await self.client.get(**http_req_kwargs)
        resp = _check_resp(resp)
        return self._post_get_cb(resp)


class SearchEngineTool(HTTPToolMeta):
    """Tool for performing web searches.

    This tool executes web searches using search engines like Google via SerpAPI.
    It can be used as a standalone function or integrated with an LLM agent.
    """

    service = conf.serpapi
    name: str = "search_engine"
    description: str = "Search the web for information on a given topic"

    def http_kwargs(self, search_query: str) -> dict:
        """Create HTTP request kwargs for search engine request.

        Args:
            search_query: The search query to execute

        Returns:
            Dictionary of HTTP request parameters
        """
        _kwargs = {
            "url": self.service.base_url,
            "params": {
                "q": search_query,
                "engine": "google",
                "api_key": self.service.api_key,
            },
        }
        return deep_merge(_kwargs, self._extra_kwargs)

    async def run(self, search_query: str) -> dict:
        """Execute a web search.

        Args:
            search_query: The search query to execute

        Returns:
            Search results from the search engine
        """
        return await self.execute_get(search_query=search_query)


class FetchPageTool(HTTPToolMeta):
    """Tool for fetching a web page's content.

    To use with text, you can configure it as follows:
    ```python
    fetch_web_tool._extra_kwargs = {"headers": {"Accept": "application/text"}}
    fetch_web_tool._post_get_cb = lambda x: x.text
    ```
    """

    service = conf.jina
    name: str = "fetch_page"
    description: str = "Fetch the full text content of a webpage by URL"

    def http_kwargs(self, url: str) -> dict:
        """Create HTTP request kwargs for page fetch request.

        Uses `Accept: application/json` to get JSON response.
        Without this header, it would return a text response.

        Args:
            url: The URL to fetch content from

        Returns:
            Dictionary of HTTP request parameters
        """
        _kwargs = {
            "url": f"{self.service.base_url}/{url}",
            "headers": {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.service.api_key}",
            },
        }
        return deep_merge(_kwargs, self._extra_kwargs)

    async def run(self, url: str) -> dict:
        """Fetch a webpage's content.

        Args:
            url: The URL to fetch content from

        Returns:
            The webpage content
        """
        return await self.execute_get(url=url)


class CachedHTTPMixin(HTTPToolMeta):
    """Mixin that adds caching to HTTPToolMeta subclasses.

    Note: This mixin should only be used with HTTPToolMeta subclasses.
    """

    _check_cache: bool = True
    _save_cache: bool = True
    _mark_from_cache: bool = True

    async def _run_with_cache(self, **kwargs) -> dict:
        """Execute a request with caching.

        This method relies on methods from HTTPToolMeta subclasses.

        Args:
            **kwargs: Parameters needed to construct the HTTP request

        Returns:
            Response data, either from cache or from a new request
        """
        http_kwargs = self.http_kwargs(**kwargs)
        req_hash = hash_str(str(http_kwargs))

        if res := get_from_cache(req_hash, callback=json.loads, check_cache=self._check_cache):
            if self._mark_from_cache:
                res["from_cache"] = req_hash
            return res

        # Rather than getting kwargs again, pass in the http_kwargs
        res = await super().execute_get(http_req_kwargs=http_kwargs)
        save_to_cache(req_hash, res, callback=json.dumps, save_cache=self._save_cache)
        return res


class CachedSearchEngineTool(SearchEngineTool, CachedHTTPMixin):
    """Search engine tool with caching functionality."""

    async def __call__(self, search_query: str) -> dict:
        """Execute a web search with caching.

        Args:
            search_query: The search query to execute

        Returns:
            Search results, either from cache or from a new request
        """
        return await self.run(search_query=search_query)

    async def run(self, search_query: str) -> dict:
        """Execute a web search with caching.

        Args:
            search_query: The search query to execute

        Returns:
            Search results, either from cache or from a new request
        """
        result = await self._run_with_cache(search_query=search_query)
        return result


class CachedFetchPageTool(FetchPageTool, CachedHTTPMixin):
    """Webpage fetching tool with caching functionality."""

    async def __call__(self, url: str) -> dict:
        """Execute a webpage fetch with caching.

        Args:
            url: The URL to fetch content from

        Returns:
            The webpage content, either from cache or from a new request
        """
        return await self.run(url=url)

    async def run(self, url: str) -> dict:
        """Fetch a webpage's content with caching.

        Args:
            url: The URL to fetch content from

        Returns:
            The webpage content, either from cache or from a new request
        """
        result = await self._run_with_cache(url=url)
        return result


class Researcher:
    """Class for managing research operations.

    This class will be implemented in the future.
    """

    pass
