import time
import asyncio

from uuid import uuid4
import reflex as rx
import httpx

from deepre import query
from deepre.provider import LLMProvider
from deepre.utils import logger


model_configs = {
    "reasoning": {
        "model_name": "deepseek-r1:70b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama-api-key",
    },
    "tool": {
        "model_name": "llama3.3:70b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama-api-key",
    },
}

reasoning_model = LLMProvider["ollama"](**model_configs["reasoning"])
tool_model = LLMProvider["ollama"](**model_configs["tool"])


class _Timestamp:
    @property
    def now(self) -> str:
        return time.strftime("%H:%M:%S")


Timestamp = _Timestamp()


class Response(rx.Base):
    response_text: str
    response_id: str


class ResearchState(rx.State):
    """State management for the research assistant."""

    user_query: str = ""
    iteration_limit: int = 1
    link_limit: int = 3
    final_report: str = ""
    process_logs: str = ""
    responses: list[Response] = []
    is_processing: bool = False

    model_configs = model_configs

    def _add_response(self, response: str):
        resp = Response(response_text=response, response_id=uuid4().hex)
        self.responses.append(resp)
        # yield rx.scroll_to(resp.response_id)

    def clear_logs(self) -> None:
        self.process_logs = ""

    async def scroll_bottom_response_cb(self):
        elem_id = self.responses[-1].response_id
        yield rx.scroll_to(elem_id)

    def update_logs(self, message: str):
        """Update process logs with timestamp."""
        timestamp = Timestamp.now
        self.process_logs += f"\n[{timestamp}] {message}"

    def _add_response(self, msg: str):
        """
        use like:

        self._add_response(msg)
        yield ResearchState.scroll_bottom_response_cb

        doesnt seem possible to make this combined in one func with reflex, or if it is,
        i cant figure out the correct pattern
        """
        resp = Response(response_text=msg, response_id=uuid4().hex)
        self.responses.append(resp)

    async def handle_submit(self):
        """Handle research submission."""
        self.is_processing = True
        self.final_report = ""
        self.process_logs = ""

        self.update_logs("Starting research process...")
        self.responses.append(f"Starting query for: {self.user_query}")

        async with httpx.AsyncClient() as client:
            # Generate initial search queries using both reasoning and tool models
            self.update_logs("Generating initial search queries...")
            yield

            # First get reasoning model's output, then extract queries using tool model
            model_resp = await query.generate_search_queries(
                model=reasoning_model,
                user_query=self.user_query,
            )

            self._add_response(model_resp)
            queries_result = await query.extract_queries_from_text(
                model=tool_model,
                user_query=model_resp,
            )
            # queries = queries_result.data
            queries = queries_result
            joined_queries = ", ".join(queries)

            if not queries:
                self.update_logs("No initial queries could be generated")
                yield
                return

            self.update_logs(f"Generated {len(queries)} initial queries: {joined_queries}")
            yield

            contexts = []
            iteration = 0

            while iteration < self.iteration_limit:
                self.update_logs(f"Starting research iteration {iteration + 1}")
                yield

                # Process search queries and collect links
                all_links = []
                for search_query in queries:
                    if len(all_links) >= self.link_limit:
                        break

                    self.responses.append(search_query)
                    self.update_logs(f"Searching for: {search_query}")
                    yield

                    links_result = await query.perform_search(
                        client,
                        search_query,
                    )
                    all_links.extend(links_result)

                all_links = all_links[: self.link_limit]  # Limit to top 10 links
                self.update_logs(f"Found {len(all_links)} links to process")
                yield

                # Process each link and extract relevant information
                iteration_contexts = []
                for link in all_links:
                    self.update_logs(f"Processing link: {link}")
                    yield

                    context = await query.process_link(
                        client=client,
                        link=link,
                        search_query=search_query,
                        user_query=self.user_query,
                        tool_model=tool_model,
                        reasoning_model=reasoning_model,
                    )

                    if context:
                        self.update_logs("Successfully extracted relevant information")
                        iteration_contexts.append(context)
                        yield
                    else:
                        self.update_logs("No useful information found in link")
                        yield

                self.update_logs(f"Extracted information from {len(iteration_contexts)} sources")
                yield

                contexts.extend(iteration_contexts)

                # Generate new queries based on current context
                new_queries_result = await query.get_new_search_queries(
                    model=tool_model,
                    user_query=self.user_query,
                    previous_queries=queries,
                    contexts=contexts,
                )
                # new_queries = new_queries_result.data
                new_queries = new_queries_result

                if not new_queries:
                    self.update_logs("No more queries needed, research complete")
                    yield
                    break

                queries = new_queries
                self.update_logs(f"Generated {len(queries)} new queries for next iteration")
                yield
                iteration += 1

            # Generate final report
            self.update_logs("Generating final research report...")
            yield

            final_report_result = await query.generate_final_report(
                model=reasoning_model,
                user_query=self.user_query,
                contexts=contexts,
            )
            self.final_report = final_report_result.data
            self.update_logs("Research process completed successfully")
