"""Provenance-preserving aliases for legacy dataset object labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar


T = TypeVar("T")


def parse_object_aliases(values: Sequence[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for value in values:
        source, separator, target = value.partition("=")
        source = source.strip()
        target = target.strip()
        if not separator or not source or not target:
            raise ValueError(f"object alias must be SOURCE=TARGET, got {value!r}")
        if source in aliases and aliases[source] != target:
            raise ValueError(f"conflicting aliases for {source!r}")
        aliases[source] = target
    return aliases


def canonicalize_named_mapping(
    values: Mapping[str, T],
    aliases: Mapping[str, str],
    *,
    expected_names: set[str] | None = None,
) -> dict[str, T]:
    canonical: dict[str, T] = {}
    for source, value in values.items():
        target = aliases.get(source, source)
        if target in canonical:
            raise ValueError(f"object aliases collide at canonical name {target!r}")
        canonical[target] = value
    if expected_names is not None and set(canonical) != expected_names:
        raise ValueError(
            f"expected canonical objects {sorted(expected_names)}, got {sorted(canonical)}"
        )
    return canonical


def canonicalize_rigid_object_state(
    state: Mapping[str, T], aliases: Mapping[str, str]
) -> dict[str, T]:
    """Return a shallow copy with only rigid-object dictionary keys renamed."""

    canonical = dict(state)
    rigid_objects = state.get("rigid_object")
    if rigid_objects is not None:
        if not isinstance(rigid_objects, Mapping):
            raise TypeError("state['rigid_object'] must be a mapping")
        canonical["rigid_object"] = canonicalize_named_mapping(rigid_objects, aliases)  # type: ignore[assignment]
    return canonical
