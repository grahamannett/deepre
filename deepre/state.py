import json
from uuid import uuid4

import httpx
import reflex as rx

from deepre import query
from deepre.prompts.message_template import get_model_configs
from deepre.provider import Agent, LLMProvider, ModelType
from deepre.utils import Timestamp, logger

model_configs = get_model_configs()

_reasoning_model: LLMProvider = LLMProvider["ollama"](**model_configs["reasoning"])
_tool_model: LLMProvider = LLMProvider["ollama"](**model_configs["tool"])


class Response(rx.Base):
    response_text: str
    response_id: str


class LLMInfo(rx.Base):
    typeof: str
    model_name: str
    base_url: str
    api_key: str

    _model_inst: ModelType = LLMProvider["ollama"](**model_configs["reasoning"])

    def update_inst(self):
        self._model_inst = LLMProvider["ollama"](
            model_name=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key,
        )


class ResearchState(rx.State):
    """State management for the research assistant."""

    user_query: str = ""
    iteration_limit: int = 1
    link_limit: int = 3
    final_report: str = ""
    process_logs: str = ""
    responses: list[Response] = []
    is_processing: bool = False

    models: dict[str, LLMInfo] = {
        "reasoning": LLMInfo(typeof="reasoning", **model_configs["reasoning"]),
        "tool": LLMInfo(typeof="tool", **model_configs["tool"]),
    }

    def on_change_model_field_setting(self, model_type: str, field: str, value: str):
        """
        this has to be on rx.State rather than LLMInfo because of reflex
        """
        setattr(self.models[model_type], field, value)

    async def on_submit_model_setting(self, model_typeof: str, data: dict):
        """
        this and change_model_field are basically the same thing, this is just the button version
        """
        pass

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
                # model=_reasoning_model,
                model=self.models["reasoning"]._model_inst,
                user_query=self.user_query,
            )

            self._add_response(model_resp)
            queries_result = await query.extract_queries_from_text(
                model=_tool_model,
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
                        tool_model=_tool_model,
                        reasoning_model=_reasoning_model,
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
                    model=_tool_model,
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
                model=_reasoning_model,
                user_query=self.user_query,
                contexts=contexts,
            )
            self.final_report = final_report_result.data
            self.update_logs("Research process completed successfully")
