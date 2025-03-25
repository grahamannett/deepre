from typing import AsyncIterator
from pydantic_ai import Agent
from pydantic_ai.models import Model
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Markdown
from rich.syntax import Syntax
from rich.text import Text


def prettier_code_blocks():
    """Make rich code blocks prettier and easier to copy."""

    class SimpleCodeBlock(CodeBlock):
        def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
            code = str(self.text).rstrip()
            yield Text(self.lexer_name, style="dim")
            yield Syntax(
                code,
                self.lexer_name,
                theme=self.theme,
                background_color="default",
                word_wrap=True,
            )
            yield Text(f"/{self.lexer_name}", style="dim")

    Markdown.elements["fence"] = SimpleCodeBlock


async def run_with_live(
    prompt: str,
    agent: Agent,
    model: Model | None = None,
    console: Console | None = None,
    live: Live | None = None,
):
    if console is None:
        console = Console()
    result_data = ""
    with Live("", console=console, vertical_overflow="visible") as live:
        async with agent.run_stream(prompt, model=model) as result:
            async for message in result.stream_text(delta=True):
                result_data += message
                live.update(Markdown(result_data))
    return result_data
