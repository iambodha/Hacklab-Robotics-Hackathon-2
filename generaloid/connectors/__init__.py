"""Connector registry: a connector is a package generaloid/connectors/<name>/ that exposes `CONNECTOR`.

    get("elrobot")                      # built-in connector by name
    get("mypkg.my_robot:CONNECTOR")     # any importable connector object (out-of-tree robots)
    available()                         # names of the built-in connectors
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from .base import Connector

_HERE = Path(__file__).resolve().parent


def available() -> list[str]:
    return sorted(m.name for m in pkgutil.iter_modules([str(_HERE)]) if m.ispkg)


def get(name: str) -> Connector:
    if ":" in name:
        mod, attr = name.split(":", 1)
        conn = getattr(importlib.import_module(mod), attr)
    else:
        if name not in available():
            raise KeyError(f"unknown connector {name!r}; built-in: {available()} (or give module:attr)")
        conn = importlib.import_module(f"{__name__}.{name}").CONNECTOR
    if not isinstance(conn, Connector):
        raise TypeError(f"{name} is not a generaloid Connector")
    return conn
