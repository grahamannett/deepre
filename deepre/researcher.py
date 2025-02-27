import functools
import json
from typing import Any, Callable, Dict, Optional

import httpx
from pydantic import BaseModel

from deepre.app_config import conf
from deepre.utils import cache_util, logger

# Constants
# CACHE_STR = "[deep_pink4]CACHE[/deep_pink4]"
CACHE_STR = "[bold green]CACHE[/bold green]"


def _cache_hit_log_make(entity_type: str, entity_name: str, hash_value: str) -> Callable[[], None]:
    """Create a function that logs a cache hit event."""

    def _log_hit():
        logger.info(
            f"{CACHE_STR} {entity_type} [cyan]Cache hit[/cyan] "
            f"for {entity_type.strip()}: [yellow]'{entity_name}'[/yellow] "
            f"[dim](hash: {hash_value})[/dim]"
        )

    return _log_hit


def _cache_save_log_make(entity_type: str, entity_name: str, hash_value: str) -> Callable[[], None]:
    """Create a function that logs a cache save event."""

    def _log_save():
        logger.info(
            f"{CACHE_STR} {entity_type} [cyan]Cache saved[/cyan] "
            f"for {entity_type.strip()}: [yellow]'{entity_name}'[/yellow] "
            f"[dim](hash: {hash_value})[/dim]"
        )

    return _log_save


def _cache_flags(
    self_check_cache: bool,
    self_save_cache: bool,
    check_cache: bool | None = None,
    save_cache: bool | None = None,
) -> tuple[bool, bool]:
    check_cache = self_check_cache if check_cache is None else check_cache
    save_cache = self_save_cache if save_cache is None else save_cache
    return check_cache, save_cache


class PageResult(BaseModel):
    """Result of a web page search."""

    url: str
    date: str = ""
    title: Optional[str] = None
    description: Optional[str] = None
    page_text: Optional[str] = None
    hash: str = ""  # Will be populated during validation

    def model_post_init(self, __context: Any) -> None:
        """Calculate the hash if not already set."""
        if not self.hash:
            self.hash = cache_util.create_url_hash(self.url, date=self.date)


class Researcher:
    """Class for performing web research operations."""

    pass


class WebServices:
    """Service class for web research operations."""

    def __init__(self):
        self.jina_api_key = conf.jina.api_key
        self.jina_base_url = conf.jina.base_url
        self.serpapi_api_key = conf.serpapi.api_key
        self.serpapi_base_url = conf.serpapi.base_url

    def _check_resp(self, resp: httpx.Response) -> bool:
        """Check if a response is valid.

        Args:
            resp: Response to check
            url: URL that was requested

        Returns:
            True if the response is valid, raises an exception otherwise
        """
        if resp.status_code != 200:
            raise Exception(f"Failed to fetch {resp.url}: {resp.status_code}")
        return True

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
        client: httpx.AsyncClient,
        url: str | None = None,
        req_kwargs: dict[str, Any] | None = None,
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

        resp = await client.get(**req_kwargs)
        self._check_resp(resp)
        return resp.text

    async def get_search_results(
        self,
        client: httpx.AsyncClient,
        search_query: str | None = None,
        req_kwargs: dict[str, Any] | None = None,
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

        resp = await client.get(**req_kwargs)
        self._check_resp(resp)

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

    async def get_page_content(
        self,
        client: httpx.AsyncClient,
        url: str,
        date: str = "",
        check_cache: Optional[bool] = None,
        save_cache: Optional[bool] = None,
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

        # Create hash for caching
        req_kwargs = self._get_page_kwargs(url)
        req_hash = self.cache_manager.create_hash(str(req_kwargs), date)

        _log_hit = _cache_hit_log_make("📄", url, req_hash)
        _log_save = _cache_save_log_make("📄", url, req_hash)

        # Check cache if requested
        if check_cache:
            cached_text = self.cache_manager.get_from_cache(req_hash, cache_dir=self.cache_dir)
            if cached_text is not None:
                _log_hit()
                return str(cached_text)

        # If not in cache or not checking cache, fetch from API
        page_text = await super().get_page_content(client, url)

        if save_cache:
            self.cache_manager.save_to_cache(req_hash, page_text, cache_dir=self.cache_dir)
            _log_save()

        return page_text

    async def get_search_results(
        self,
        client: httpx.AsyncClient,
        search_query: str,
        check_cache: bool | None = None,
        save_cache: bool | None = None,
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

        req_kwargs = self._get_search_kwargs(search_query)
        req_hash = self.cache_manager.create_hash(str(req_kwargs))

        _log_hit = _cache_hit_log_make("🔍", search_query, req_hash)
        _log_save = _cache_save_log_make("🔍", search_query, req_hash)

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
        resp_json = await super().get_search_results(client, req_kwargs=req_kwargs)

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


class WebServicesTool(WebServicesWithCache):
    async def get_search_results_as_page_results(
        self,
        client: httpx.AsyncClient,
        page: PageResult,
    ) -> list[PageResult]:
        """Convert search results for a query into a list of PageResult objects.

        Args:
            client: HTTP client to use for the request
            page: PageResult object containing the search query in its url field

        Returns:
            List of PageResult objects created from the search results
        """
        try:
            resp_json = await super().get_search_results(client, page.url)

            # Use list comprehension for cleaner code and better performance
            return [
                PageResult(
                    url=item["link"],
                    date=item.get("date", ""),  # Default to empty string to avoid validation errors
                    title=item.get("title"),
                    description=item.get("snippet"),
                )
                for item in resp_json.get("organic_results", [])
                if "link" in item
            ]

        except Exception as e:
            logger.error(f"Error fetching search results as PageResults: {str(e)}")
            return []  # Return empty list on error for graceful degradation
