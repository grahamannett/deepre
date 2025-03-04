from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings

mock_user_system_prompt = """You are being used to mimic the user for input and output queries.
Your responses are typically one to two sentences unless it makes sense to provide a longer response.
For everything you say, you should provide a response that would be similar to how a user testing this system would respond.
The original query is: <prompt>{original_query}</prompt>.
Assume all following messages are messages that would be presented to the user and you are to imitate their response.
"""


def make_mock_user_agent(
    model: OpenAIModel, original_query: str, model_settings: ModelSettings | None = None, **kwargs
) -> Agent:
    mock_agent = Agent(
        model,
        model_settings=model_settings,
        system_prompt=mock_user_system_prompt.format(original_query=original_query),
        **kwargs,
    )

    return mock_agent
