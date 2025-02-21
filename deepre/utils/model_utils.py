from dataclasses import dataclass, field
from typing import Type

from pydantic_ai.models import Model


@dataclass(init=False)
class ModelNameMixin:
    _model_name: str = field(repr=True)


def get_model_class(class_name: str, model_base: Type[Model]) -> Type[ModelNameMixin]:
    ModelCls = type(class_name, (ModelNameMixin, model_base), {})
    return ModelCls


"""
replace the ModelClients stuff with just this. initial idea was just have more extensible way to handle json data


class ModelClients:
    reasoning_model_config = model_configs["reasoning"]
    tool_model_config = model_configs["tool"]

    reasoning_client = LLMProvider["ollama"](**reasoning_model_config)
    tool_client = LLMProvider["ollama"](**tool_model_config)

    @property
    def reasoning_json(self):
        return get_client_json(self.reasoning_model_config)

    @reasoning_json.setter
    def reasoning_json(self, value):
        set_client_json(self, "reasoning_model_config", value)
        self.reasoning_client = LLMProvider["ollama"](**self.reasoning_model_config)

    @property
    def tool_json(self):
        return get_client_json(self.tool_model_config)

    @tool_json.setter
    def tool_json(self, value):
        set_client_json(self, "tool_model_config", value)
        self.tool_client = LLMProvider["ollama"](**self.tool_model_config)

"""


class ModelClients:
    """
    was trying to figure out best way to hold the model configs that can be changeable.
    i dont think this will work with reflex since the state needs to be a part of the rx.State
    # model_clients = ModelClients()
    """

    clients = {}
    configs: dict[str, dict] = {}

    def __init__(self, model_configs):
        self.configs = {k: v for k, v in model_configs.items()}
        self._setup_client("reasoning")
        self._setup_client("tool")

    def _setup_client(self, client_name: str):
        self.clients[client_name] = LLMProvider["ollama"](**self.configs[client_name])

    def _set_json_data(self, client_name: str, data: dict):
        self.configs[client_name] = data
        self._setup_client(client_name)

    @property
    def reasoning_json(self):
        return get_client_json(self.configs["reasoning"])

    @reasoning_json.setter
    def reasoning_json(self, value):
        if json_data := validate_json(value):
            self._set_json_data("reasoning", json_data)

    @property
    def tool_json(self):
        return get_client_json(self.configs["tool"])

    @tool_json.setter
    def tool_json(self, value):
        if json_data := validate_json(value):
            self._set_json_data("tool", json_data)

    def validate_model_json(self, model_data: str):
        logger.info(f"should validate: {model_data=}")
