import os
from dataclasses import dataclass

import httpx
from pydantic_ai import Agent, Tool
from pydantic_ai.models.openai import OpenAIModel


@dataclass
class FetchPageTool:
    client: httpx.AsyncClient
    jina_api_key: str
    base_url: str = "https://r.jina.ai"

    async def __call__(self, url: str) -> str:
        res = await self.client.get(
            url=f"{self.base_url}/{url}",
            headers={"Authorization": f"Bearer {self.jina_api_key}"},
        )
        return res.text


@dataclass
class SearchTool:
    client: httpx.AsyncClient
    serpapi_api_key: str
    base_url: str = "https://serpapi.com/search"

    async def __call__(self, query: str) -> dict:
        res = await self.client.get(
            url=self.base_url,
            params={
                "q": query,
                "engine": "google",
                "api_key": self.serpapi_api_key,
            },
        )

        return res.json()


async def main():
    async with httpx.AsyncClient() as client:
        fetch_page_tool = FetchPageTool(client=client, jina_api_key=os.environ["JINA_API_KEY"])
        page_results = await fetch_page_tool(
            url="https://aimagazine.com/ai-applications/deep-research-inside-openais-new-analysis-tool"
        )

        search_tool = Tool(
            SearchTool(client=client, serpapi_api_key=os.environ["SERPAPI_API_KEY"]).__call__,
            name="search_tool",
            description="Searches SERP for the given query and returns the results.",
            takes_ctx=False,
        )
        agent = Agent(
            model=OpenAIModel(
                model_name="llama3.2:latest",
                base_url="http://localhost:11434/v1",
            ),
            tools=[search_tool],
        )
        results = await agent.run(user_prompt="how does deep researcher work")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
