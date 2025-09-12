from typing import Callable, Type
from src.models.base import BaseNightingaleModel

MODEL_REGISTRY = {}
def register_model(name: str) -> Callable:
    """
    Register a model.
    """
    def decorator(cls: Type[BaseNightingaleModel]) -> Type[BaseNightingaleModel]:
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator