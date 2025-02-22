import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from functools import cache

import httpx
from pydantic import BaseModel, computed_field
from pydantic_ai import Agent, RunContext, capture_run_messages
from pydantic_ai.models.openai import OpenAIModel
from rich.prompt import Prompt

from deepre.app_config import conf
from deepre.utils import logger

base_q = "How does openai deep research work?"
base_s_q = "openai deep research explanation"

cache_dir = ".cache"
CACHE_STR = "[deep_pink4]CACHE[/deep_pink4]"


@cache
def _cache_dir_check(cache_dir: str = cache_dir):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)


def _print_msgs(*msgs):
    for i, msg in enumerate(msgs):
        logger.debug(f"[{i}]\n\t{msg}")


def _hash_str(link: str, date: str = None, hash_len: int = 10, **kwargs) -> str:
    # 10 seems like short enough to be unique for our purposes, no idea if this is true
    hash_str = hashlib.md5(f"{link}{date}".encode()).hexdigest()
    return hash_str[:hash_len]


def _cache_check(
    hash: str,
    check_cache: bool = True,
    cache_dir: str = cache_dir,
    _cb: callable = lambda x: x,
) -> str | None:
    """Check if the hash is in the cache."""
    if check_cache is False:
        return None

    hash_file = f"{cache_dir}/{hash}"
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            return _cb(f.read())

    return None


def _cache_save(
    hash: str,
    text: str | dict,
    save_cache: bool = True,
    cache_dir: str = cache_dir,
    _cb: callable = lambda x: x,
) -> bool:
    """Save the text to the cache."""
    if save_cache is False:
        return False

    with open(f"{cache_dir}/{hash}", "w") as f:
        f.write(_cb(text))

    return True


@dataclass
class AgentSetup:
    model_name: str = "ishumilin/deepseek-r1-coder-tools:70b"
    alt_model_name: str = "llama3.2:latest-extended"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"


class PageResult(BaseModel):
    """Result of a web page search."""

    url: str
    date: str | None = ""
    title: str | None = None
    description: str | None = None

    page_text: str | None = None

    @computed_field
    @property
    def hash(self) -> str:
        return _hash_str(self.url, date=self.date)


class Failed(BaseModel):
    """Failed to generate queries."""

    ...


@dataclass
class DeepResearchDeps:
    """Dependency class for deep research operations."""

    http_client: httpx.AsyncClient
    original_query: str

    page_results: list[PageResult] = field(default_factory=list)

    def contains(self, url: str) -> bool:
        return any(url == page.url for page in self.page_results)

    current_date: str = datetime.now().strftime("%Y-%m-%d")
    mock_user: bool = True


model_settings = {"temperature": 0.0}

model = OpenAIModel(
    model_name=AgentSetup.model_name,
    base_url=AgentSetup.base_url,
    api_key=AgentSetup.api_key,
)

mock_agent = Agent(model, model_settings=model_settings)
research_agent = Agent(model, deps_type=DeepResearchDeps, model_settings=model_settings)

serp_agent = Agent(
    OpenAIModel(
        model_name=AgentSetup.alt_model_name,
        base_url=AgentSetup.base_url,
        api_key=AgentSetup.api_key,
    ),
    deps_type=DeepResearchDeps,
    model_settings=model_settings,
)


@serp_agent.system_prompt
async def serp_agent_system_prompt(ctx: RunContext[DeepResearchDeps]) -> str:
    """Generate the system prompt for the agent."""
    prompt = """For a given search term use the `get_search_results` tool with the search query to fetch a list of `PageResult`."""
    return prompt


@mock_agent.system_prompt
async def mock_system_prompt(ctx: RunContext[DeepResearchDeps]) -> str:
    """Generate the system prompt for the agent."""
    prompt = f"""
You are being used to mimic the user for input and output queries. Your responses are typically one to two sentences unless it makes sense to provide a longer response.
For everything you say, you should provide a response that would be similar to how a user testing this system would respond.,
The original query is: <prompt>{ctx.deps.original_query}</prompt>.
Assume all following messages are messages that would be presented to the user and you are to imitate their response.
"""
    return prompt


@research_agent.system_prompt
async def research_agent_system_prompt(ctx: RunContext[DeepResearchDeps]) -> str:
    """Generate the system prompt for the agent."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
You are an agent in an AI system that serves as a research assistant, designed to help users research various topics using advanced online search and data analysis capabilities.
Your responses are typically one to two sentences unless the users request requires detailed reasoning or long-form outputs, and you avoid using emojis unless explicitly asked to.

Current date: {current_date}

Your primary purpose is to assist with tasks requiring extensive research using the `start_research_task` and `get_user_response` functions, accessing only publicly available internet data and locally uploaded files, not websites requiring authentication.

## Guidelines for Using the `start_research_task`

1. **Ask the user for more details before starting research**
   - Before initiating research with `start_research_task`, use `get_user_response` to ask for more details unless the user provides exceptionally detailed information, ensuring you have enough to proceed effectively.
   - Examples of when to ask clarifying questions with `get_user_response`:
       - For “Research snowboards,” ask about budget, terrain type, skill level, or brand preferences.
       - For “Which washing machine should I buy?” ask about budget, capacity needs, or brand preferences.
       - For “Plan a European vacation,” ask about travel dates, preferred countries, activities, and budget.
       - For “Research stocks to buy,” ask about risk tolerance, investment goals, or preferred industries.
       - For “Outline a marketing strategy for my small business,” ask about business type, target audience, or budget.
       - For “Find an elegant restaurant for a celebratory dinner,” ask about location, dietary preferences, or party size.
       - For “Give a lit review of biology developments,” ask about subfields, time range, or depth.
       - For “Find the best place for a data center,” ask about location requirements, size, or power needs.
   - Keep questions concise and relevant using `get_user_response`, avoiding repetition of what the user already said, and use a friendly tone like “Could you share more about X?” instead of “I need more info.”
   - If the request is exceptionally detailed, skip clarification and use `start_research_task` directly."""
    return prompt


@research_agent.tool
async def get_user_response(ctx: RunContext[DeepResearchDeps], query: str) -> str:
    if ctx.deps.mock_user:
        # simulate the response from the user
        res = await mock_agent.run(query, deps=ctx.deps, result_type=str)
        resp = res.data
        print(f">>For this query:\n\t`{query}`\ngenerated this mock response:`{resp}`||\n")
    else:
        resp = Prompt.ask(query + "\n>> ")
    return resp


# async def start_research_task(ctx: RunContext[DeepResearchDeps], serp_query: str):
@research_agent.tool
async def start_research_task(ctx: RunContext[DeepResearchDeps], serp_query: list[str]):
    """Start a research task using the list of gathered terms user query and research intensity."""

    for query in serp_query:
        logger.info(f"Running search for {query}")
        page_results = await serp_agent.run(query, deps=ctx.deps, result_type=list[PageResult])
        ctx.deps.page_results.extend(page_results)


@research_agent.tool
async def get_final_report(ctx: RunContext[DeepResearchDeps]) -> str:
    resp = await research_agent.run(
        "Generate a final report using the collected information",
        deps=ctx.deps,
        result_type=str,
    )
    return resp.data


@serp_agent.tool
async def get_search_results(ctx: RunContext[DeepResearchDeps], search_query: str) -> list[PageResult]:
    """
    Search the web and get the page text for the given search term
    """
    client = ctx.deps.http_client
    page_results = ctx.deps.page_results

    page_list = await perform_search(client=client, search_query=search_query)

    for page in page_list:
        # check if the page has already been fetched
        if ctx.deps.contains(page):
            logger.info(f"Page already fetched for {page.url}, skipping")
            continue

        logger.info(f'getting page text for "{page.url}"')
        if (page_result := await fetch_page_result(client=client, page=page)) is not None:
            page_results.append(page_result)
            logger.info(f"Added page result for {page.url}")

    logger.info(f"collected: {len(page_results)} results for {search_query}")
    return page_results


async def perform_search(
    client: httpx.AsyncClient,
    search_query: str,
    check_cache: bool = True,
    save_cache: bool = True,
) -> list[PageResult]:
    """
    Perform a search with the provided search query and return a list of PageResult objects.
    For dev purposes, a saved example of the client.get is in `tests/fixtures/search_results.json`
    """
    results = []

    req_kwargs = {
        "url": conf.serpapi.base_url,
        "params": {
            "q": search_query,
            "engine": "google",
            "api_key": conf.serpapi.api_key,
        },
    }

    search_hash = _hash_str(str(req_kwargs))

    if (resp_json := _cache_check(search_hash, check_cache=check_cache, _cb=json.loads)) is not None:
        logger.info(f"{CACHE_STR}|search-found| {search_hash} for {search_query}")
    else:
        if (resp := await client.get(**req_kwargs)).status_code != 200:
            logger.error(f"Error fetching search results for {search_query}")
            return results

        resp_json = resp.json()

        if _cache_save(search_hash, resp_json, save_cache=save_cache, _cb=json.dumps):
            logger.info(f"{CACHE_STR}|search-saved| {search_hash} for {search_query}")

    for item in resp_json.get("organic_results", []):
        if "link" not in item:
            logger.warning(f"Missing link in search result item: {item}")
            continue

        page = PageResult(
            url=item["link"],
            date=item.get("date"),
            title=item.get("title"),
            description=item.get("snippet"),
        )
        results.append(page)

    return results


async def fetch_page_result(
    client: httpx.AsyncClient,
    page: PageResult,
    check_cache: bool = True,
    save_cache: bool = True,
) -> PageResult | None:
    """
    Fetch the text content of a webpage using the provided URL.
    cache the result based on the hash of the url and the date in the metadata to save on jina calls
    """

    get_kwargs = {
        "url": f"{conf.jina.base_url}{page.url}",
        "headers": {"Authorization": f"Bearer {conf.jina.api_key}"},
    }

    if (text := _cache_check(page.hash, check_cache=check_cache)) is not None:
        logger.info(f"{CACHE_STR}|page-found| {page.hash} for {page.url}")
    else:
        resp = await client.get(**get_kwargs)
        # can get 200 and result still is messed up with this api, deal with later
        if resp.status_code != 200:
            logger.error(f"Error fetching page text for {page.url}")
            return None

        text = resp.text
        if save_cache:
            _cache_save(page.hash, text, save_cache=save_cache)
            logger.info(f"{CACHE_STR}|page-saved| {page.hash} for {page.url}")

    page.page_text = text
    return page


async def main(query: str = base_q):
    async with httpx.AsyncClient() as client:
        deps = DeepResearchDeps(http_client=client, original_query=query)

        with capture_run_messages() as messages:
            result = await research_agent.run(query, deps=deps)
            msgs = result.all_messages()


async def search(query: str = base_s_q, original_query: str = base_q):
    async with httpx.AsyncClient() as client:
        deps = DeepResearchDeps(http_client=client, original_query=original_query)

        with capture_run_messages() as messages:
            result = await serp_agent.run(query, deps=deps, result_type=list[PageResult])
            msgs = result.all_messages()

        logger.info(f"Finished lé search, got: {result.data}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Deep Research CLI")
    parser.add_argument("--mode", "-m", choices=["main", "search"], default="main")
    parser.add_argument("--query", "-q", type=str, help="Query to research/search")

    args = parser.parse_args()

    # Set default query based on mode
    if args.query is None:
        args.query = base_s_q if args.mode == "search" else base_q

    return args


if __name__ == "__main__":
    import asyncio

    args = parse_args()

    asyncio.run(
        {
            "main": main,
            "search": search,
        }[args.mode](query=args.query)
    )
