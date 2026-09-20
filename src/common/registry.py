"""Name -> model factory, so train.py never imports a model directly.

Owner: TV 2. Model owners (TV 1, TV 3) register their own class in their own file,
which keeps everyone out of each other's diffs.
"""
from __future__ import annotations

from typing import Callable

import torch.nn as nn

_MODELS: dict[str, Callable[..., nn.Module]] = {}


def register(name: str):
    """Decorator: @register("cnn") above a model class."""
    def wrapper(factory):
        if name in _MODELS:
            raise KeyError(f"model '{name}' already registered")
        _MODELS[name] = factory
        return factory
    return wrapper


def build_model(config: dict) -> nn.Module:
    """config['model'] = {'name': 'cnn', ...} -> instantiated model."""
    params = dict(config["model"])
    name = params.pop("name")
    if name not in _MODELS:
        raise KeyError(f"unknown model '{name}'. Registered: {sorted(_MODELS)}")
    return _MODELS[name](**params)


def available() -> list[str]:
    return sorted(_MODELS)
