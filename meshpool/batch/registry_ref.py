"""Structural type for what the engines need from the node registry.

Using a Protocol keeps the workload engines decoupled from the concrete
control-plane ``NodeRegistry`` (and avoids import cycles).
"""
from __future__ import annotations

from typing import Protocol

from ..common.models import Node


class RegistryRef(Protocol):
    def schedulable_nodes(self) -> list[Node]: ...
    def online(self) -> list[Node]: ...
    def list(self) -> list[Node]: ...
    def get(self, node_id: str) -> Node | None: ...
