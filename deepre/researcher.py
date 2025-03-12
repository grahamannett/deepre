import json
from typing import Any, Callable

import httpx
from pydantic_ai import Tool

from deepre.app_config import EndpointExternalServiceConfig, conf
from deepre.utils.cache_utils import get_from_cache, hash_str, save_to_cache


def default_json_resp(resp: httpx.Response) -> dict:
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


def deep_merge(base: dict, updates: dict | None = None, **kwargs) -> dict:
    """Recursively merge dictionaries.

    The updates dictionary values take precedence over base values,
    except for nested dictionaries which are merged recursively.

    Args:
        base: Base dictionary
        updates: Dictionary with values to update/extend the base dict

    Returns:
        Merged dictionary
    """
    if (not updates) and (not kwargs):
        return base
    elif (not updates) and kwargs:  # handle when kwargs used but no updates dict
        updates = kwargs
    elif updates and kwargs:  # handle when both updates and kwargs are passed in
        updates = deep_merge(updates, kwargs)

    result = base.copy()

    for key, value in updates.items():
        # If both are dicts, merge them recursively
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:  # Otherwise, override the value
            result[key] = value

    return result


class ToolMixin:
    name: str
    description: str

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        # this is to allow for subclasses to have their own post init, e.g. for validating
        self.__post_init__()

    def __post_init__(self) -> None:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    def __eq__(self, other) -> bool:
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.__dict__.items())))

    async def __call__(self, *args, **kwargs) -> Any:
        # helper func to allow for usage/debugging.  prefer to use `run`
        return await self.run(*args, **kwargs)

    async def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Must be implemented by subclasses")

    def as_tool(self, **kwargs) -> Tool:
        tool_kwargs = {"takes_ctx": False, "name": self.name, "description": self.description} | kwargs
        return Tool(function=self.run, **tool_kwargs)


class HTTPToolMeta(ToolMixin):
    service: EndpointExternalServiceConfig
    # either set http_kwargs on init or
    # allow client to be set after init, makes it easier to use with pydantic-ai or for threading
    client: httpx.AsyncClient | None = None

    # dont add http_kwargs to this as it will result in the subclasses not having the correct dict
    _post_get_cb: Callable = staticmethod(default_json_resp)  # Changed to staticmethod to avoid passing self
    _extra_kwargs: dict | None = None  # allow setting of extra_kwargs that will merge into the http_kwargs

    def http_kwargs(self, **kwargs) -> dict:
        """
        allow for changes to service to be reflected in http_kwargs

        in the case where you want to change the base http_kwargs, you should override the `http_kwargs` method
        """

        raise NotImplementedError("Must be implemented PER service class")

    async def execute_get(self, *args, **kwargs) -> dict:
        """Execute an HTTP request and process the response."""
        if not self.client:
            raise ValueError("Must set client on init or after")

        if "http_req_kwargs" in kwargs:
            http_req_kwargs = kwargs.pop("http_req_kwargs")
        else:
            http_req_kwargs = self.http_kwargs(*args, **kwargs)

        resp = await self.client.get(**http_req_kwargs)
        resp = _check_resp(resp)
        return self._post_get_cb(resp)


# @dataclass
class SearchEngineTool(HTTPToolMeta):
    """Tool for performing web searches.

    This tool executes web searches using search engines like Google via SerpAPI.
    It can be used as a standalone function or integrated with an LLM agent.
    """

    service = conf.serpapi

    name: str = "search_engine"
    description: str = "Search the web for information on a given topic"

    def http_kwargs(self, search_query: str) -> dict:
        _kwargs = {
            "url": self.service.base_url,
            "params": {
                "q": search_query,
                "engine": "google",
                "api_key": self.service.api_key,
            },
        }
        # possibly move this to base class, or use decorator on http_kwargs
        return deep_merge(_kwargs, self._extra_kwargs)

    async def run(self, search_query: str) -> dict:
        return await self.execute_get(search_query=search_query)


class FetchPageTool(HTTPToolMeta):
    """
    Tool for fetching a web page's content.

    To use with text, can do something like

    fetch_web_tool._extra_kwargs = {"headers": {"Accept": "application/text"}}
    fetch_web_tool._post_get_cb = lambda x: x.text
    """

    service = conf.jina
    name: str = "fetch_page"
    description: str = "Fetch the full text content of a webpage by URL"

    def http_kwargs(self, url: str) -> dict:
        _kwargs = {
            "url": f"{self.service.base_url}/{url}",
            "headers": {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.service.api_key}",
            },
        }

        return deep_merge(_kwargs, self._extra_kwargs)

    async def run(self, url: str) -> dict:
        return await self.execute_get(url=url)


class CachedHTTPMixin(HTTPToolMeta):
    """Mixin that adds caching to HTTPToolMeta subclasses.  This ISNT a subclass of HTTPToolMeta,
    but defining the type correctly is a lot of bloat
    Note: This mixin should only be used with HTTPToolMeta subclasses.
    """

    _check_cache: bool = True
    _save_cache: bool = True
    _mark_from_cache: bool = True

    async def _run_with_cache(self, **kwargs) -> dict:
        """Execute a request with caching.
        This method relies on methods from HTTPToolMeta subclasses.
        """
        http_kwargs = self.http_kwargs(**kwargs)
        req_hash = hash_str(str(http_kwargs))

        if res := get_from_cache(req_hash, callback=json.loads, check_cache=self._check_cache):
            if self._mark_from_cache:
                res["from_cache"] = req_hash
            return res
        # rather than getting kwargs again, can pass in the http_kwargs
        res = await super().execute_get(http_req_kwargs=http_kwargs)
        save_to_cache(req_hash, res, callback=json.dumps, save_cache=self._save_cache)
        return res


class CachedSearchEngineTool(SearchEngineTool, CachedHTTPMixin):
    async def run(self, search_query: str) -> dict:
        result = await self._run_with_cache(search_query=search_query)
        return result


class CachedFetchPageTool(FetchPageTool, CachedHTTPMixin):
    async def run(self, url: str) -> dict:
        result = await self._run_with_cache(url=url)
        return result


class Researcher:
    pass
