"""core.lazy: PEP 562 lazy re-exports so importing a light submodule doesn't import the whole package (and torch)."""
from __future__ import annotations

import importlib
from typing import Dict, List, Tuple


def install_lazy_exports(package: str, mapping: Dict[str, Tuple[str, str]], namespace: dict) -> None:
    """mapping: exported name -> (module path, attribute name). Installs __getattr__/__dir__ into ``namespace``."""

    def __getattr__(name: str):
        try:
            module, attr = mapping[name]
        except KeyError:
            raise AttributeError(f"module '{package}' has no attribute '{name}'") from None
        value = getattr(importlib.import_module(module), attr)
        namespace[name] = value  # cache so __getattr__ runs once per name
        return value

    def __dir__() -> List[str]:
        return sorted(set(namespace) | set(mapping))

    namespace["__getattr__"] = __getattr__
    namespace["__dir__"] = __dir__
