import functools
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import httpx
from pydantic import BaseModel, TypeAdapter

from deepre.app_config import conf
from deepre.utils import cache_util, logger


def _check_resp(resp: httpx.Response) -> None:
    """Check if a response is valid.

    Args:
        resp: Response to check
        url: URL that was requested

    Returns:
        True if the response is valid, raises an exception otherwise
    """
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch {resp.url}: {resp.status_code}")


class PageResult(BaseModel):
    """Result of a web page search."""

    url: str
    title: str
    date: str = ""
    description: Optional[str] = None
    page_text: Optional[str] = None
    hash: Optional[str] = None  # Will be populated during validation

    def model_post_init(self, __context: Any) -> None:
        """Calculate the hash if not already set."""
        if not self.hash:
            self.hash = cache_util.create_url_hash(self.url, date=self.date)


class Failed(BaseModel):
    """Failed to generate queries."""

    ...


class Researcher:
    """Class for performing web research operations.

    This class serves as the main entry point for creating and using research tools.
    It provides factory methods to create tools that can be used with LLM agents.
    """

    def __init__(self, web_services=None, use_cache=True):
        """Initialize the researcher with web services.

        Args:
            web_services: Web services class instance to use for API calls
            use_cache: Whether to use caching for API calls
        """
        self.client = httpx.AsyncClient()

        if web_services:
            self.web_services = web_services
        elif use_cache:
            self.web_services = WebServicesWithCache(client=self.client)
        else:
            self.web_services = WebServices(client=self.client)

    @property
    def tools(self):
        """Return a list of tools that can be used with this researcher.

        Returns:
            List of tool classes that can be instantiated and used with models/agents
        """
        return [SearchQueryTool, FetchPageTool, ResearchTool]

    def create_tool(self, tool_type: str, **kwargs):
        """Create and return a tool instance.

        Args:
            tool_type: Type of tool to create ('search', 'fetch', or 'research')
            **kwargs: Additional arguments to pass to the tool constructor

        Returns:
            An instance of the requested tool

        Raises:
            ValueError: If the tool_type is not recognized
        """
        tools = {"search": SearchQueryTool, "fetch": FetchPageTool, "research": ResearchTool}

        if tool_type not in tools:
            raise ValueError(f"Unknown tool type: {tool_type}. Available types: {', '.join(tools.keys())}")

        # Add web_services if not provided
        if "web_services" not in kwargs:
            kwargs["web_services"] = self.web_services

        return tools[tool_type](**kwargs)

    def get_pydantic_ai_tools(self, include=None):
        """Get a list of tools wrapped for use with pydantic-ai.

        Args:
            include: Optional list of tool types to include ('search', 'fetch', 'research')
                    If None, all tools are included

        Returns:
            List of Tool objects that can be passed to an Agent
        """
        try:
            from pydantic_ai import Tool
        except ImportError:
            raise ImportError(
                "pydantic-ai is required to use this method. Install with 'pip install pydantic-ai'"
            )

        available_tools = {
            "search": self.create_tool("search"),
            "fetch": self.create_tool("fetch"),
            "research": self.create_tool("research"),
        }

        if include:
            # Filter to only requested tools
            tool_instances = {k: v for k, v in available_tools.items() if k in include}
        else:
            tool_instances = available_tools

        # Wrap in pydantic-ai Tool objects
        return [
            Tool(tool.__call__, name=f"{name}_tool", description=tool.description)
            for name, tool in tool_instances.items()
        ]


@dataclass
class FetchPageTool:
    """Tool for fetching a web page's content.

    This tool retrieves the full text of a webpage using a service like Jina AI.
    It can be used as a standalone function or integrated with an LLM agent.
    """

    web_services: Optional[WebServices] = None
    api_key: str = conf.jina.api_key
    base_url: str = conf.jina.base_url
    description: str = "Fetch the full text content of a webpage by URL"

    def __post_init__(self):
        """Initialize web services if not provided."""
        if self.web_services is None:
            self.client = httpx.AsyncClient()
            self.web_services = WebServices(client=self.client)

    async def __call__(self, url: str) -> str:
        """Fetch the full text content of a webpage.

        Args:
            url: The URL of the page to fetch

        Returns:
            The full text content of the webpage
        """
        if self.web_services:
            return await self.web_services.get_page_content(url=url)

        resp = await self.client.get(
            url=f"{self.base_url}/{url}",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        _check_resp(resp)
        return resp.text


@dataclass
class SearchQueryTool:
    """Tool for performing web searches.

    This tool executes web searches using search engines like Google via SerpAPI.
    It can be used as a standalone function or integrated with an LLM agent.
    """

    web_services: Optional[WebServices] = None
    api_key: str = conf.serpapi.api_key
    base_url: str = conf.serpapi.base_url
    description: str = "Search the web for information on a given topic"

    def __post_init__(self):
        """Initialize web services if not provided."""
        if self.web_services is None:
            self.client = httpx.AsyncClient()
            self.web_services = WebServices(client=self.client)

    async def __call__(self, query: str) -> list[PageResult]:
        """Perform a web search and return structured results.

        Args:
            query: The search query to execute

        Returns:
            A list of PageResult objects containing search results
        """
        if self.web_services:
            results = await self.web_services.get_search_results(search_query=query)
            return self._parse_results(results)

        resp = await self.client.get(
            url=self.base_url,
            params={
                "q": query,
                "api_key": self.api_key,
                "engine": "google",
            },
        )
        _check_resp(resp)
        return self._parse_results(resp.json())

    def _parse_results(self, results: dict) -> list[PageResult]:
        """Parse the raw search results into PageResult objects.

        Args:
            results: Raw search results JSON from the API

        Returns:
            List of PageResult objects
        """
        page_results = []
        for item in results.get("organic_results", []):
            if "link" not in item:
                continue

            page = PageResult(
                url=item["link"],
                title=item.get("title", ""),
                date=item.get("date", ""),
                description=item.get("snippet"),
            )
            page_results.append(page)

        return page_results


@dataclass
class ResearchTool:
    """Tool for comprehensive web research on a topic.

    This tool combines searching and content fetching to provide complete
    research on a topic, extracting the most relevant information.
    """

    web_services: Optional[WebServicesWithCache] = None
    search_tool: Optional[SearchQueryTool] = None
    fetch_tool: Optional[FetchPageTool] = None
    description: str = "Research a topic by searching the web and analyzing the results"
    max_results: int = 3

    def __post_init__(self):
        """Initialize required components if not provided."""
        if self.web_services is None:
            self.client = httpx.AsyncClient()
            self.web_services = WebServicesWithCache(client=self.client)

        if self.search_tool is None:
            self.search_tool = SearchQueryTool(web_services=self.web_services)

        if self.fetch_tool is None:
            self.fetch_tool = FetchPageTool(web_services=self.web_services)

    async def __call__(self, query: str) -> list[PageResult]:
        """Perform comprehensive research on a topic.

        This method:
        1. Searches for information on the query
        2. Fetches the full text of the top results
        3. Returns structured results with full page content

        Args:
            query: The research query/topic

        Returns:
            List of PageResult objects with full page content
        """
        # Get search results
        page_results = await self.search_tool(query)

        # Limit to max_results
        page_results = page_results[: self.max_results]

        # Fetch full text for each result
        for page in page_results:
            try:
                page_text = await self.fetch_tool(page.url)
                page.page_text = page_text
            except Exception as e:
                logger.error(f"Error fetching page {page.url}: {e}")

        return page_results


class WebServices:
    """Service class for web research operations."""

    def __init__(self, client: httpx.AsyncClient):
        self.jina_api_key = conf.jina.api_key
        self.jina_base_url = conf.jina.base_url
        self.serpapi_api_key = conf.serpapi.api_key
        self.serpapi_base_url = conf.serpapi.base_url
        self.client = client

    @functools.cache
    def _get_search_kwargs(self, search_query: str) -> dict[str, Any]:
        """Perform a search with the provided search query and return the search results.

        Args:
            search_query: The search query to perform

        Returns:
            Dictionary of request arguments
        """
        return {
            "url": self.serpapi_base_url,
            "params": {
                "q": search_query,
                "engine": "google",
                "api_key": self.serpapi_api_key,
            },
        }

    @functools.cache
    def _get_page_kwargs(self, url: str) -> dict[str, Any]:
        """Fetch the text content of a webpage.

        Args:
            url: URL to fetch

        Returns:
            Dictionary of request arguments
        """
        return {
            "url": f"{self.jina_base_url}{url}",
            "headers": {"Authorization": f"Bearer {self.jina_api_key}"},
        }

    async def get_page_content(
        self,
        url: str | None = None,
        req_kwargs: dict[str, Any] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> str:
        """Fetch the text content of a webpage.

        Use with pydantic-ai like:
        > Agent(..., tools=[fetch_page]) or Agent(..., tools=[Tool(fetch_page, takes_ctx=False),])

        Args:
            client: HTTP client to use for the request
            url: URL to fetch
            req_kwargs: Request kwargs to use for the request
        Returns:
            Text content of the webpage

        Raises:
            Exception: If the request fails
        """
        if not any([url, req_kwargs]) or all([url, req_kwargs]):
            raise ValueError("Exactly one of url or req_kwargs must be provided")

        if not req_kwargs:
            req_kwargs = self._get_page_kwargs(url)

        client = client or self.client

        resp = await client.get(**req_kwargs)
        _check_resp(resp)
        return resp.text

    async def get_search_results(
        self,
        search_query: str | None = None,
        req_kwargs: dict[str, Any] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> Dict[str, Any]:
        """Perform a search using either a search_query string or req_kwargs dictionary.

        Exactly one of search_query or req_kwargs must be provided.

        Args:
            client: HTTP client to use for the request
            search_query: The search query to perform (if req_kwargs not provided)
            req_kwargs: The request kwargs to use directly (if search_query not provided)

        Returns:
            JSON response containing search results

        Raises:
            ValueError: If both or neither of search_query and req_kwargs are provided
            Exception: If the request fails
        """
        if not any([req_kwargs, search_query]) or all([req_kwargs, search_query]):
            raise ValueError("Exactly one of req_kwargs or search_query must be provided")

        if not req_kwargs:
            req_kwargs = self._get_search_kwargs(search_query)

        client = client or self.client

        resp = await client.get(**req_kwargs)
        _check_resp(resp)

        return resp.json()


class WebServicesWithCache(WebServices):
    def __init__(
        self,
        cache_dir: str = cache_util.DEFAULT_CACHE_DIR,
        check_cache: bool = True,
        save_cache: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cache_dir = cache_dir
        self.check_cache = check_cache
        self.save_cache = save_cache

        # Ensure cache directory exists
        cache_util.ensure_cache_dir(cache_dir)

        # Create cache manager instance
        self.cache_manager = cache_util.CacheManager()

    def _cache_flags(self, check: bool | None, save: bool | None) -> tuple[bool, bool]:
        check = self.check_cache if check is None else check
        save = self.save_cache if save is None else save
        return check, save

    def _cache_log_make(
        self,
        entity_type: str,
        entity_name: str,
        hash_value: str,
        cache_action: str,
    ) -> Callable[[], None]:
        """Create a function that logs a cache event."""
        CACHE_STR = "[bold green]CACHE[/bold green]"

        msg = (
            f"{CACHE_STR} {entity_type} [cyan]Cache {cache_action}[/cyan] "
            f"for {entity_type.strip()}: [yellow]'{entity_name}'[/yellow] "
            f"[dim](hash: {hash_value})[/dim]"
        )
        return lambda: logger.info(msg)

    async def get_page_content(
        self,
        url: str,
        date: str = "",
        check_cache: Optional[bool] = None,
        save_cache: Optional[bool] = None,
        client: httpx.AsyncClient | None = None,
    ) -> str:
        """Fetch the text content of a webpage with caching.

        Args:
            client: HTTP client to use for the request
            url: URL to fetch
            date: Optional date string to include in the hash
            check_cache: Whether to check the cache (overrides instance default)
            save_cache: Whether to save to the cache (overrides instance default)

        Returns:
            Text content of the webpage

        Raises:
            Exception: If the request fails
        """
        check_cache, save_cache = self._cache_flags(check_cache, save_cache)
        client = client or self.client

        # Create hash for caching
        req_kwargs = self._get_page_kwargs(url)
        req_hash = self.cache_manager.create_hash(str(req_kwargs), date)

        _log_hit = self._cache_log_make("📄", url, req_hash, "hit")
        _log_save = self._cache_log_make("📄", url, req_hash, "save")

        # Check cache if requested
        if check_cache:
            cached_text = self.cache_manager.get_from_cache(req_hash, cache_dir=self.cache_dir)
            if cached_text is not None:
                _log_hit()
                return str(cached_text)

        # If not in cache or not checking cache, fetch from API
        page_text = await super().get_page_content(req_kwargs=req_kwargs, client=client)

        if save_cache:
            self.cache_manager.save_to_cache(req_hash, page_text, cache_dir=self.cache_dir)
            _log_save()

        return page_text

    async def get_search_results(
        self,
        search_query: str,
        check_cache: bool | None = None,
        save_cache: bool | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, Any]:
        """Perform a search with the provided search query and return the search results.

        This version uses caching to avoid repeated API calls.

        Args:
            client: HTTP client to use for the request
            search_query: The search query to perform
            check_cache: Whether to check the cache (overrides instance default)
            save_cache: Whether to save to the cache (overrides instance default)

        Returns:
            JSON response containing search results

        Raises:
            Exception: If the request fails
        """
        check_cache, save_cache = self._cache_flags(check_cache, save_cache)
        client = client or self.client

        req_kwargs = self._get_search_kwargs(search_query)
        req_hash = self.cache_manager.create_hash(str(req_kwargs))

        _log_hit = self._cache_log_make("🔍", search_query, req_hash, "hit")
        _log_save = self._cache_log_make("🔍", search_query, req_hash, "save")

        # Check cache if requested
        if check_cache:
            cached_result = self.cache_manager.get_from_cache(
                req_hash,
                callback=json.loads,
                cache_dir=self.cache_dir,
            )

            if cached_result is not None:
                _log_hit()
                return cached_result if isinstance(cached_result, dict) else {}

        # Not in cache or not checking cache, fetch from API
        resp_json = await super().get_search_results(req_kwargs=req_kwargs, client=client)

        # Save to cache if requested
        if save_cache:
            self.cache_manager.save_to_cache(
                req_hash,
                resp_json,
                callback=json.dumps,
                cache_dir=self.cache_dir,
            )
            _log_save()

        return resp_json
