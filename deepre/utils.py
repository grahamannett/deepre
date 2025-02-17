from dataclasses import dataclass, field
from typing import Type, TypeAlias

from pydantic_ai.models import Model
from reflex.utils import console

logger: TypeAlias = console


@dataclass(init=False)
class ModelNameMixin:
    _model_name: str = field(repr=True)


def get_model_class(class_name: str, model_base: Type[Model]) -> Type[ModelNameMixin]:
    ModelCls = type(class_name, (ModelNameMixin, model_base), {})
    return ModelCls
