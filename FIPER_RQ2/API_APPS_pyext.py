"""Small compatibility utilities used by older generated-program tooling."""

from __future__ import annotations

import functools
import inspect
import sys
import types
from typing import Any, Callable


__all__ = [
    "RuntimeModule",
    "annotate",
    "assign",
    "call_if_main",
    "compare_and_swap",
    "is_main",
    "run_main",
    "safe_unpack",
    "switch",
    "tail_recurse",
]


class _RuntimeModule:
    """Create an importable module from source text or named objects."""

    def __call__(self, *args: Any, **kwargs: Any) -> types.ModuleType:
        return self.from_objects(*args, **kwargs)

    @staticmethod
    def from_objects(name: str, docstring: str = "", **objects: Any) -> types.ModuleType:
        module = types.ModuleType(name, docstring)
        module.__dict__.update(objects)
        module.__file__ = "<runtime-module>"
        sys.modules[name] = module
        return module

    @staticmethod
    def from_string(name: str, source: str, docstring: str = "") -> types.ModuleType:
        namespace: dict[str, Any] = {"__name__": name, "__doc__": docstring}
        exec(compile(source, f"<{name}>", "exec"), namespace, namespace)
        objects = {key: value for key, value in namespace.items() if not key.startswith("__")}
        return _RuntimeModule.from_objects(name, docstring, **objects)


RuntimeModule = _RuntimeModule()


class _Switch:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.matched = False
        self.stopped = False

    def __enter__(self) -> "_Switch":
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def __call__(self, *values: Any) -> bool:
        if self.stopped or self.matched:
            return False
        self.matched = self.value in values
        return self.matched

    def default(self) -> bool:
        return not self.matched and not self.stopped

    def quit(self) -> None:
        self.stopped = True


def switch(value: Any) -> _Switch:
    return _Switch(value)


def tail_recurse(spec: Callable[..., bool] | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Convert a simple tail-recursive function into an iterative wrapper."""
    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return function(*args, **kwargs)

        return wrapped

    del spec
    return decorate


def annotate(*annotations: Any, **named_annotations: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(function)
        names = list(signature.parameters)
        function.__annotations__.update(dict(zip(names, annotations)))
        return_annotation = named_annotations.pop("ret", None)
        if return_annotation is not None:
            named_annotations["return"] = return_annotation
        function.__annotations__.update(named_annotations)
        return function

    return decorate


def safe_unpack(sequence: Any, length: int, fill: Any = None) -> Any:
    values = list(sequence)
    values = values[:length] + [fill] * max(0, length - len(values))
    return type(sequence)(values) if not isinstance(sequence, tuple) else tuple(values)


def assign(variable_name: str, value: Any) -> Any:
    caller_globals = inspect.currentframe().f_back.f_globals  # type: ignore[union-attr]
    if "." not in variable_name:
        caller_globals[variable_name] = value
        return value
    parts = variable_name.split(".")
    target = caller_globals[parts[0]]
    for part in parts[1:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)
    return value


def compare_and_swap(mapping: dict[Any, Any], key: Any, expected: Any, replacement: Any) -> bool:
    if mapping.get(key) != expected:
        return False
    mapping[key] = replacement
    return True


def is_main(frame_depth: int = 1) -> bool:
    return inspect.stack()[frame_depth].frame.f_globals.get("__name__") == "__main__"


def call_if_main(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return function(*args, **kwargs) if is_main(2) else None


def run_main(function: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    result = call_if_main(function, *args, **kwargs)
    if result is not None:
        raise SystemExit(result)
