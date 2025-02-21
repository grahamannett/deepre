import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, capture_run_messages
from pydantic_ai.models.openai import OpenAIModel
from rich.prompt import Prompt

from deepre.app_config import conf
from deepre.utils import logger

base_q = "How does openai deep research work?"
base_s_q = "openai deep research explanation"


cache_dir = ".cache"


def _print_msgs(*msgs):
    for i, msg in enumerate(msgs):
        logger.debug(f"[{i}]\n\t{msg}")


def _hash_str(link: str, date: str = None, **kwargs) -> str:
    hash_str = hashlib.md5(f"{link}{date}".encode()).hexdigest()
    # 10 seems like short enough to be unique for our purposes, no idea if this is true
    hash_str = hash_str[:10]
    return hash_str


def _check_cache(hash: str, cache_dir: str = cache_dir) -> str | None:
    """Check if the hash is in the cache."""
    hash_file = f"{cache_dir}/{hash}"
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            return f.read()
    return None


def _save_cache(hash: str, text: str, cache_dir: str = cache_dir) -> None:
    """Save the text to the cache."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    with open(f"{cache_dir}/{hash}", "w") as f:
        f.write(text)


@dataclass
class AgentSetup:
    model_name: str = "ishumilin/deepseek-r1-coder-tools:70b"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"

    """
    might be helpful for
    """


"""
just made some helper DTO classes for helping me to design the system
"""


class ResearchQuery(BaseModel):
    user_query: str
    need_clarification: bool | None = None
    # followup_queries: list["ResearchQuery"] | None = None


class PageMetadata(BaseModel):
    """Metadata for a web page."""

    url: str
    hash: str
    title: str | None = None
    description: str | None = None


class PageResult(BaseModel):
    metadata: PageMetadata | str
    page_text: str
    # extracted_text: str | None = None


class Failed(BaseModel):
    """Failed to generate queries."""

    ...


def _url_in(url: str, result: PageResult | str) -> bool:
    """Check if the URL is in the result."""
    if isinstance(result, PageResult):
        return url == result.metadata.url
    return url == result


@dataclass
class DeepResearchDeps:
    """Dependency class for deep research operations."""

    http_client: httpx.AsyncClient
    original_query: str

    page_results: list[PageResult] = field(default_factory=list)

    def contains(self, url: str) -> bool:
        return any(_url_in(url, result) for result in self.page_results)

    #
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
        model_name="llama3.2:latest",
        base_url=AgentSetup.base_url,
        api_key=AgentSetup.api_key,
    ),
    deps_type=DeepResearchDeps,
    model_settings=model_settings,
)


@serp_agent.system_prompt
async def serp_agent_system_prompt(ctx: RunContext[DeepResearchDeps]) -> str:
    """Generate the system prompt for the agent."""
    # You are an agent in an AI system that serves as a research assistant.
    # You are the part of the system that is focused on SERP and can search terms to get url's which you will then extract relevant information from.
    #       to find webpage url's.
    # For a webpage url, you are to use the `fetch_page` tool to fetch the web page result.  Once the page has been fetched, you are to extract the relevant text from the page.
    prompt = """For a given search term use the `get_search_results` tool with the search query to fetch a list of webpage results"""
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


# 2. **Strongly bias towards using the `start_research_task`**
#    - Treat every non-trivial question the user asks as a research task and use `clarify_with_text` or `start_research_task` accordingly.
#    - Trivial questions, such as exchanging greetings or asking for simple text transformations, can be handled directly without research."""


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


@research_agent.tool
async def start_research_task(ctx: RunContext[DeepResearchDeps], serp_query: str):
    """Start a research task using the user query and research intensity."""
    breakpoint()
    # first gather search results
    resp = await serp_agent.run(serp_query, deps=ctx.deps, result_type=list[PageResult])
    # coalese the results
    breakpoint()


@serp_agent.tool
async def get_search_results(ctx: RunContext[DeepResearchDeps], search_query: str) -> list[PageResult]:
    """
    Search the web and get the page text for the given search term
    """
    client = ctx.deps.http_client
    page_results = ctx.deps.page_results

    metadata_list = await perform_search(client, search_query)

    for metadata in metadata_list:
        # check if the page has already been fetched
        if ctx.deps.contains(metadata):
            logger.info(f"Page already fetched for {metadata.url}, skipping")
            continue

        logger.info(f'getting page text for "{metadata.url}"')
        if (page_result := await fetch_page_result(client, metadata)) is not None:
            logger.info(f"Added page result for {metadata.url}")
            page_results.append(page_result)

    return page_results


async def perform_search(client: httpx.AsyncClient, search_query: str) -> list[PageMetadata]:
    """
    Perform a search with the provided search query and return a list of PageMetadata objects.

    For dev purposes, a saved example of the client.get is in `tests/fixtures/search_results.json`

    Possibly could have this `yield` a result so i search the result but idk if that will impact reflex
    """
    results = []

    resp = await client.get(
        url=conf.serpapi.base_url,
        params={
            "q": search_query,
            "engine": "google",
            "api_key": conf.serpapi.api_key,
        },
    )

    if resp.status_code == 200:
        resp_json = resp.json()
        for item in resp_json.get("organic_results", []):
            if "link" not in item:
                logger.warning(f"Missing link in search result item: {item}")
            else:
                results.append(
                    PageMetadata(
                        url=item["link"],
                        hash=_hash_str(**item),
                        title=item["title"],
                        description=item["snippet"],
                    )
                )

    return results


async def fetch_page_result(
    client: httpx.AsyncClient,
    metadata: str | PageMetadata,
    check_cache: bool = True,
    save_cache: bool = True,
) -> PageResult | None:
    """
    Fetch the text content of a webpage using the provided URL.

    cache the result based on the hash of the url and the date in the metadata to save on jina calls


    Using jina api to fetch page text for now, returns markdown of the page.
    """
    url = metadata.url if isinstance(metadata, PageMetadata) else metadata

    if check_cache and (text := _check_cache(metadata.hash)) is not None:
        logger.info(f"Found cached result for {metadata.url}")

    else:
        resp = await client.get(
            url=f"{conf.jina.base_url}{url}",
            headers={"Authorization": f"Bearer {conf.jina.api_key}"},
        )
        # can get 200 and result still is messed up with this api, deal with later
        if resp.status_code != 200:
            logger.error(f"Error fetching page text: {resp.text}")
            return None

        text = resp.text
        if save_cache:
            _save_cache(metadata.hash, text)
            logger.info(f"Cache {metadata.hash[:]} saved for {metadata.url}")

    return PageResult(metadata=metadata, page_text=text)


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
            result = await serp_agent.run(query, deps=deps)
            msgs = result.all_messages()

        _print_msgs(*msgs)
        logger.info(f"Finished lé search, got these messages, {msgs}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Deep Research CLI")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["main", "search"],
        default="main",
        help="Run mode: main or search (default: main)",
    )

    # Add query argument without default - we'll set it based on mode
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Query to research",
    )

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
