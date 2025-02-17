from dataclasses import asdict
from typing import Dict

import pytest
import tomllib

from deepre.prompts.message_template import MessageTemplate, template_toml_file


def test_read_template():
    # Test loading template from TOML
    with open(template_toml_file, "rb") as f:
        template_data = tomllib.load(f)

    template_name = list(template_data.keys())[0]

    template = MessageTemplate.use_toml(template_name)

    # Verify template was loaded with correct attributes
    assert isinstance(template, MessageTemplate)

    # Test template formatting works
    formatted_prompt = template.get_system_prompt()
    assert formatted_prompt == template.system_prompt

    # Test user content formatting with required parameters
    formatted_content = template.get_user_content(
        user_query="What is quantum computing?", num_queries=3
    )

    assert "What is quantum computing?" in formatted_content


@pytest.mark.parametrize(
    "template_dict,expected_user_content",
    [
        (
            {
                "system_prompt": "You are a test research assistant.",
                "user_content": "Test Query: {user_query} with {extra_param}",
            },
            "Test Query: What is Python? with additional info",
        ),
        (
            {
                "system_prompt": "You are another test assistant.",
                "user_content": "{user_query} - {extra_param}",
            },
            "What is Python? - additional info",
        ),
        (
            {
                "system_prompt": "You are a minimal assistant.",
                "user_content": "{user_query}",
            },
            "What is Python?",
        ),
    ],
)
def test_research_assistant_template(template_dict: Dict[str, str], expected_user_content: str):
    """Test ResearchAssistantTemplate with different configurations."""
    template = MessageTemplate(**template_dict)

    # Test creation from dict
    assert template.system_prompt == template_dict["system_prompt"]
    assert template.user_content == template_dict["user_content"]

    # Test conversion to dict
    template_as_dict = asdict(template)
    assert template_as_dict["system_prompt"] == template.system_prompt
    assert template_as_dict["user_content"] == template.user_content

    # Test formatting with parameters if the template expects them
    try:
        user_content = template.get_user_content(
            user_query="What is Python?",
            extra_param="additional info",
        )
        assert user_content == expected_user_content
    except KeyError as e:
        # If the template doesn't use extra_param, it should raise KeyError
        assert "extra_param" not in template.user_content


@pytest.mark.parametrize(
    "template_dict,missing_param",
    [
        (
            {
                "system_prompt": "Test assistant",
                "user_content": "Query: {user_query} with {required_param}",
            },
            "required_param",
        ),
        (
            {
                "system_prompt": "Another assistant",
                "user_content": "{user_query} needs {param1} and {param2}",
            },
            "param1",
        ),
    ],
)
def test_missing_parameters(template_dict: Dict[str, str], missing_param: str):
    """Test that missing required parameters raise KeyError."""
    template = MessageTemplate(**template_dict)
    with pytest.raises(KeyError) as exc_info:
        template.get_user_content(user_query="What is Python?")
    assert missing_param in str(exc_info.value)


def test_default_values():
    """Test default values when creating template without parameters."""
    template = MessageTemplate()
    assert template.system_prompt == "You are a helpful and precise research assistant."
    assert template.user_content == "User Query: {user_query}"
