from datetime import datetime
from typing import TypeAlias, TypeVar

import httpx
from pydantic_ai.result import RunResult

from deepre.app_config import conf
from deepre.prompts import MessageTemplate
from deepre.provider import Agent, ModelType

T = TypeVar("T")
ModelRunResult: TypeAlias = RunResult[T] | T


def _response[T](resp: RunResult[T], return_data: bool = True) -> RunResult[T] | T:
    """
    Process the response from a model run and return either the data or the full RunResult.

    This function allows for flexible handling of model outputs, enabling easy switching
    between returning just the data or the full RunResult object. This flexibility can be
    useful for debugging, logging, or when additional metadata from the run is needed.

    Args:
        resp (RunResult[T]): The result object from a model run.
        return_data (bool, optional): If True, return only the data. If False, return the full RunResult. Defaults to True.

    Returns:
        RunResult[T] | T: Either the data (T) or the full RunResult[T], depending on the return_data flag.
    """
    if return_data:
        return resp.data
    return resp


async def fetch_webpage_text(
    client: httpx.AsyncClient,
    url: str,
) -> str:
    """Fetch webpage text using Jina API.

    Args:
        client (httpx.AsyncClient): The HTTP client to use for the request
        url (str): The URL of the webpage to fetch

    Returns:
        str: The text content of the webpage if successful, empty string otherwise

    Note:
        Uses Jina API with authentication for fetching webpage content.
        Returns empty string on any errors or non-200 status codes.
    """
    full_url = f"{conf.jina.base_url}{url}"
    headers = {
        "Authorization": f"Bearer {conf.jina.api_key}",
    }
    try:
        resp = await client.get(url=full_url, headers=headers)
        return resp.text if resp.status_code == 200 else ""
    except Exception:
        return ""


async def perform_search(client: httpx.AsyncClient, query: str) -> list[str]:
    """Perform web search using SERPAPI.

    Args:
        client (httpx.AsyncClient): The HTTP client to use for the request
        query (str): The search query to execute

    Returns:
        list[str]: List of URLs from the search results' organic links.
                  Returns empty list if the request fails or no results found.
    """
    params = {
        "q": query,
        "api_key": conf.serpapi.api_key,
        "engine": "google",
    }
    try:
        resp = await client.get(conf.serpapi.base_url, params=params)
        if resp.status_code == 200:
            results = resp.json()
            return [item.get("link") for item in results.get("organic_results", []) if "link" in item]
        return []
    except Exception:
        return []


async def process_link(
    client: httpx.AsyncClient,
    link: str,
    search_query: str,
    user_query: str,
    tool_model: ModelType,
    reasoning_model: ModelType | None = None,
    force_is_useful: bool = False,
):
    """Process a single link and extract relevant information.

    Args:
        client: httpx client
        link: URL to process
        search_query: Original search query
        user_query: User's query
        tool_model: Main model to use for processing
        reasoning_model: Optional model to use for initial context extraction
        force_is_useful: If True, skips the usefulness check
    """
    page_text = await fetch_webpage_text(client=client, url=link)
    if not page_text:
        return None
    # Check if page is useful unless forced
    is_useful = force_is_useful or await is_page_useful(
        model=tool_model,
        user_query=user_query,
        webpage_text=page_text,
    )

    if is_useful:
        # Use reasoning model for initial context extraction if provided
        context = await extract_relevant_context(
            model=reasoning_model or tool_model,
            user_query=user_query,
            search_query=search_query,
            page_text=page_text,
        )

        # If using reasoning model, process the context with main model
        if reasoning_model:
            context_str = getattr(context, "data", context)
            context = await delegate_extract_relevant_context(
                model=tool_model,  # Use main model for final processing
                model_output=context_str,
            )
        return context

    return None


async def is_page_useful(
    model: ModelType,
    user_query: str,
    webpage_text: str,
) -> RunResult[bool] | bool:
    """Determine if a webpage's content is relevant to the user's query.

    Args:
        model (ModelType): The language model to use for relevance assessment
        user_query (str): The original user query
        webpage_text (str): The text content of the webpage to evaluate

    Returns:
        RunResult[bool] | bool: True if the page is deemed useful for the query, False otherwise
    """
    template = MessageTemplate.use_toml(key="page_useful")

    system_prompt = template.get_system_prompt()
    user_prompt = template.get_user_content(user_query=user_query, webpage_text=webpage_text)

    agent = Agent(model=model, system_prompt=system_prompt)
    resp = await agent.run(user_prompt=user_prompt, result_type=bool)
    return _response(resp)


async def generate_search_queries(
    model: ModelType,
    user_query: str,
    num_queries: int = 5,
    now: str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    result_type: type | None = None,
) -> RunResult[list[str]] | list[str]:
    """Generate multiple search queries based on a user query using an LLM.

    Args:
        model (ModelType): The language model to use for query generation
        user_query (str): The original user query to expand upon
        num_queries (int, optional): Number of search queries to generate. Defaults to 5
        now (str, optional): Current timestamp in ISO format. Defaults to current time

    Returns:
        RunResult[list[str]] | list[str]: Generated search queries based on the user's original query
    """
    template = MessageTemplate.use_toml(key="search_query")
    template.update(user_query=user_query, now=now, num_queries=num_queries)
    system_prompt = template.get_system_prompt()
    user_prompt = template.get_user_content()

    agent = Agent(model=model, system_prompt=system_prompt)
    resp = await agent.run(user_prompt=user_prompt, result_type=result_type)
    return _response(resp)


async def extract_queries_from_text(
    model: ModelType,
    user_query: str,
) -> RunResult[list[str]] | list[str]:
    """Extract potential follow-up queries from a given text using an LLM.

    Args:
        model (ModelType): The language model to use for query extraction
        user_query (str): The original user query to base follow-up queries on

    Returns:
        RunResult[list[str]] | list[str]: List of extracted follow-up queries
    """
    template = MessageTemplate.use_toml(key="extract_query")
    template.update(user_query=user_query)
    system_prompt = template.get_system_prompt()
    user_prompt = template.get_user_content()

    # Doesnt seem to work with result_type=list[...QueryType...], so using list[str] for now
    agent = Agent(model=model, system_prompt=system_prompt)
    resp = await agent.run(user_prompt=user_prompt, result_type=list[str])
    return _response(resp)


async def extract_relevant_context(
    model: ModelType,
    user_query: str,
    search_query: str,
    page_text: str,
) -> RunResult[str] | str:
    """Extract relevant information from webpage content based on user and search queries.

    Args:
        model (ModelType): The language model to use for context extraction
        user_query (str): The original user query
        search_query (str): The search query that led to this page
        page_text (str): The text content of the webpage

    Returns:
        RunResult[str] | str: Extracted relevant context from the page
    """
    template = MessageTemplate.use_toml(key="context_extraction")  # alt is `extract_relevant_context`
    template.update(user_query=user_query, search_query=search_query, page_text=page_text)
    system_prompt = template.get_system_prompt()
    user_prompt = template.get_user_content()

    agent = Agent(model, system_prompt=system_prompt)
    resp = await agent.run(user_prompt=user_prompt)
    return _response(resp)


async def delegate_extract_relevant_context(
    model: ModelType,
    model_output: str,
) -> RunResult[str] | str:
    """Delegate the extraction of relevant context from model output to another model instance.

    Args:
        model (ModelType): The language model to use for context extraction
        model_output (str): The output from a previous model run to extract context from

    Returns:
        RunResult[str] | str: Refined or extracted context from the model output
    """
    template = MessageTemplate.use_toml(key="delegate_context_extraction")
    template.update(model_output=model_output)
    system_prompt = template.get_system_prompt()
    user_prompt = template.get_user_content()
    agent = Agent(model, system_prompt=system_prompt)
    resp = await agent.run(user_prompt=user_prompt)
    return _response(resp)


async def get_new_search_queries(
    model: ModelType,
    user_query: str,
    previous_queries: list[str],
    contexts: list[str],
) -> RunResult[list[str]] | list[str]:
    """Generate new search queries based on the user query, previous queries, and contexts.

    Args:
        model (ModelType): The language model to use for query generation.
        user_query (str): The original user query.
        previous_queries (list[str]): List of previously generated queries.
        contexts (list[str]): List of relevant contexts from previous searches.

    Returns:
        RunResult[list[str]] | list[str]: A list of new search queries
    """
    template = MessageTemplate.use_toml(key="new_queries")
    template.update(
        user_query=user_query,
        previous_queries=previous_queries,
        contexts=contexts,
        # prompt_purpose="`Return a list of new queries for the `Query` based on the `Contexts` and `Previous Queries`.",
    )
    system_prompt = template.get_system_prompt()
    user_prompt = template.get_user_content()

    agent = Agent(model, system_prompt=system_prompt)
    resp = await agent.run(user_prompt=user_prompt, result_type=list[str])
    return _response(resp)


async def generate_final_report(
    model: ModelType,
    user_query: str,
    contexts: list[str],
) -> RunResult[str] | str:
    """Generate a final research report based on the user query and context.

    Args:
        model (ModelType): The language model to use for report generation.
        user_query (str): The original user query.
        contexts (list[str]): List of contexts to use for report generation.

    Returns:
        RunResult[str] | str: The generated research report
    """
    template = MessageTemplate.use_toml(key="final_report")
    template.update(user_query=user_query, contexts=contexts)
    system_prompt = template.get_system_prompt()
    user_prompt = template.get_user_content()

    agent = Agent(model, system_prompt=system_prompt)
    resp = await agent.run(user_prompt=user_prompt)
    return _response(resp)
