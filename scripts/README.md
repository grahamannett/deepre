# SERP Agent

The SERP (Search Engine Results Page) Agent is a tool for searching the web and retrieving page content. It can be used as a standalone tool or integrated into larger research workflows.

## Features

- Search the web using Google search engine
- Retrieve and cache search results
- Fetch and cache page content
- Limit the number of results
- Output results to a file (JSON format)

## Usage

```bash
python scripts/serp-example.py --query "your search query" --limit 5
```

### Command-line Arguments

- `--query`, `-q`: Search query (required)
- `--limit`, `-l`: Maximum number of results (default: 5)
- `--cache-dir`, `-c`: Cache directory (default: `.cache`)
- `--no-cache`: Disable cache
- `--output`, `-o`: Output file for results (JSON format)
- `--model`, `-m`: Model to use (default: `llama3.2:latest-extended`)
- `--base-url`, `-b`: Base URL for the model API (default: `http://localhost:11434/v1`)
- `--api-key`, `-k`: API key for the model (default: `ollama`)

## How It Works

The SERP Agent uses the `Researcher` class to perform web searches and fetch page content. It follows these steps:

1. Parse command-line arguments
2. Create a `SerpAgent` instance with the specified configuration
3. Perform a search using the `search` method
4. Fetch page content for each search result
5. Limit the results to the specified number
6. Print the results to the console
7. Save the results to a file if specified

## Example Output

```
Search Results for: pydantic-ai agent pattern
================================================================================
Found 3 results

1. Agents - PydanticAI
   URL: https://ai.pydantic.dev/agents/
   Description: Agents are PydanticAI's primary interface for interacting with LLMs. In some use cases a single Agent...

2. PydanticAI
   URL: https://ai.pydantic.dev/
   Description: PydanticAI is a Python agent framework designed to make it less painful to build production grade ap...

3. Pydantic AI and AI Agents: A Primer - Medium
   URL: https://medium.com/@contact.av.rh/pydantic-ai-and-ai-agents-a-primer-f03545b7ac43
   Description: Pydantic AI extends these capabilities, giving developers a streamlined way to define prompts, parse...
```

## Integration

The SERP Agent can be integrated into larger research workflows by importing the `SerpAgent` class and using its methods directly:

```python
from scripts.serp_example import SerpAgent, SerpAgentConfig
import httpx
import asyncio

async def main():
    # Create agent config
    config = SerpAgentConfig(
        cache_dir=".cache",
        result_limit=5,
    )

    # Create agent
    agent = SerpAgent(config)

    # Perform search
    async with httpx.AsyncClient() as client:
        results = await agent.search(
            query="your search query",
            client=client,
            check_cache=True,
            save_cache=True,
            limit=5
        )

        # Process results
        for page in results:
            print(f"Title: {page.title}")
            print(f"URL: {page.url}")
            print(f"Description: {page.description}")
            print(f"Content: {page.page_text[:100]}...")
            print()

if __name__ == "__main__":
    asyncio.run(main())
```
