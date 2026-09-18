"""Checkpoint state (de)serialization with an explicit type allow-list.

This is the ONLY place in `flow_engine` allowed to turn a stored dict back
into a custom object. It replaces two previously-duplicated reflection-based
reconstruction paths (blind `importlib.import_module` + auto-instantiate +
`setattr` off data taken straight from the checkpoint store) with a
registry the developer opts into explicitly.

Only `InMemoryCheckpointer` skips this module entirely (it keeps live
Python objects via `deepcopy`, never crossing a serialization boundary).
`RedisCheckpointer`/`SQLCheckpointer` persist to an external store, so a
tampered or foreign checkpoint payload must never be able to name an
arbitrary importable class — every custom type has to be registered ahead
of time via `FlowEngine(types=[...])` / `flow.register_type(cls)`.

Fails closed in both directions: an unregistered type raises on save (you
cannot even write it to a checkpoint) and on load (a stored tag naming a
type this process hasn't registered is rejected, never silently degraded
to a raw dict).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from pydantic import BaseModel as PydanticBaseModel

from llmfy.exception.llmfy_exception import CheckpointDeserializationException

_TYPE_TAG = "__type__"
_DATA_KEY = "data"


class TypeRegistry:
    """Explicit allow-list of custom types that may cross the checkpoint
    (de)serialization boundary.

    Only Pydantic `BaseModel` subclasses and `@dataclass` types are
    accepted — both have a safe, well-defined dump/validate pair, so
    reconstruction never needs constructor-signature guessing or blind
    `setattr`.
    """

    def __init__(self, types: list[type] | None = None):
        self._types: dict[str, type] = {}
        for cls in types or []:
            self.register(cls)

    def register(self, cls: type) -> None:
        if not (
            isinstance(cls, type)
            and (issubclass(cls, PydanticBaseModel) or dataclasses.is_dataclass(cls))
        ):
            raise CheckpointDeserializationException(
                f"Cannot register {cls!r} for checkpointing: only Pydantic "
                "BaseModel subclasses and @dataclass types are supported.",
                type_name=getattr(cls, "__qualname__", str(cls)),
            )
        self._types[qualname(cls)] = cls

    def resolve(self, name: str) -> type | None:
        return self._types.get(name)


def qualname(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def serialize_state(state: dict[str, Any], registry: TypeRegistry) -> dict[str, Any]:
    """Serialize a workflow state dict into an all-JSON-native dict.

    Raises `CheckpointDeserializationException` if a value's type isn't
    JSON-native and isn't registered in `registry`.
    """
    return {key: _serialize_value(value, registry, key) for key, value in state.items()}


def deserialize_state(raw: dict[str, Any], registry: TypeRegistry) -> dict[str, Any]:
    """Reconstruct a workflow state dict from its serialized form.

    Raises `CheckpointDeserializationException` if the payload tags a type
    that isn't registered in `registry`.
    """
    return {key: _deserialize_value(value, registry, key) for key, value in raw.items()}


def _serialize_value(value: Any, registry: TypeRegistry, field_name: str) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, list):
        return [_serialize_value(item, registry, field_name) for item in value]

    if isinstance(value, dict):
        return {k: _serialize_value(v, registry, field_name) for k, v in value.items()}

    if isinstance(value, PydanticBaseModel):
        name = qualname(type(value))
        _require_registered(registry, name, field_name)
        return {_TYPE_TAG: name, _DATA_KEY: value.model_dump(mode="json")}

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        name = qualname(type(value))
        _require_registered(registry, name, field_name)
        data = {
            f.name: _serialize_value(getattr(value, f.name), registry, field_name)
            for f in dataclasses.fields(value)
        }
        return {_TYPE_TAG: name, _DATA_KEY: data}

    raise CheckpointDeserializationException(
        f"Cannot serialize value of type '{qualname(type(value))}' for field "
        f"'{field_name}': only JSON-native values and types registered via "
        "FlowEngine(types=[...]) / flow.register_type(...) are supported.",
        type_name=qualname(type(value)),
        field_name=field_name,
    )


def _deserialize_value(value: Any, registry: TypeRegistry, field_name: str) -> Any:
    if isinstance(value, dict):
        if _TYPE_TAG in value and _DATA_KEY in value:
            name = value[_TYPE_TAG]
            cls = registry.resolve(name)
            if cls is None:
                raise CheckpointDeserializationException(
                    f"Checkpoint field '{field_name}' references type '{name}' "
                    "which is not registered. Call FlowEngine(types=[...]) or "
                    "flow.register_type(...) before loading this checkpoint.",
                    type_name=name,
                    field_name=field_name,
                )
            raw_data = value[_DATA_KEY]
            if issubclass(cls, PydanticBaseModel):
                return cls.model_validate(raw_data)
            data = {
                k: _deserialize_value(v, registry, field_name) for k, v in raw_data.items()
            }
            return cls(**data)
        return {k: _deserialize_value(v, registry, field_name) for k, v in value.items()}

    if isinstance(value, list):
        return [_deserialize_value(item, registry, field_name) for item in value]

    return value


def _require_registered(registry: TypeRegistry, name: str, field_name: str) -> None:
    if registry.resolve(name) is None:
        raise CheckpointDeserializationException(
            f"Type '{name}' is not registered for checkpointing (field "
            f"'{field_name}'). Call FlowEngine(types=[...]) or "
            "flow.register_type(...) first.",
            type_name=name,
            field_name=field_name,
        )
