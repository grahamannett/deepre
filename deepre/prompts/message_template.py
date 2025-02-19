from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepre.app_config import conf
from deepre.utils import read_toml_file

template_toml_file = conf.tomls_dir / "templates.toml"
model_config_file = conf.tomls_dir / conf.model_configs_file


def get_model_configs() -> dict[str, Any]:
    """
    Load and cache model configurations from TOML file.

    Returns:
        dict[str, Any]: Model configuration dictionary
    """
    return read_toml_file(model_config_file)


@dataclass
class MessageTemplate:
    """
    A template for generating system prompts and user content for AI interactions.

    Attributes:
        system_prompt (str): The system prompt template.
        user_content (str): The user content template.
    """

    system_prompt: str  # = "You are a helpful research assistant."
    user_content: str  # = "User Query: {user_query}"

    template_values: dict[str, Any]

    system = property(lambda self: self.get_system_prompt())
    user = property(lambda self: self.get_user_content())

    def __init__(self, **kwargs) -> None:
        self.template_values = {}

        for k, v in kwargs.items():
            setattr(self, k, v)

    def update(self, **kwargs) -> None:
        self.template_values.update(kwargs)

    def get_system_prompt(self, **kwargs: Any) -> str:
        """
        Format the system prompt with the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments to format the system prompt.

        Returns:
            str: The formatted system prompt.
        """
        return self.system_prompt.format(**{**self.template_values, **kwargs})

    def get_user_content(self, **kwargs: Any) -> str:
        """
        Format the user content with the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments to format the user content.

        Returns:
            str: The formatted user content.
        """
        return self.user_content.format(**{**self.template_values, **kwargs})

    @classmethod
    def use_toml(
        cls,
        key: str,
        toml_file: str | Path = template_toml_file,
    ) -> "MessageTemplate":
        """
        Load a template from a TOML file.

        Args:
            key (str): The key to access the template in the TOML file.
            toml_file (str | Path): The path to the TOML file. Defaults to template_toml_file.

        Returns:
            MessageTemplate: An instance of MessageTemplate with data loaded from the TOML file.

        Raises:
            KeyError: If the specified key is not found in the TOML file.
            tomllib.TOMLDecodeError: If there's an error decoding the TOML file.
        """

        data = read_toml_file(toml_file)

        try:
            template_data = data[key]
            return cls(**template_data)
        except KeyError as e:
            raise KeyError(f"Template key '{key}' not found in TOML file.") from e
