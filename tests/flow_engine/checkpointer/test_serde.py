"""Unit tests for llmfy/flow_engine/checkpointer/serde.py.

This module is the sole trust boundary for turning checkpoint dicts back
into custom objects — every test here is either a round-trip proof or a
fail-closed proof (unregistered/tampered types must be rejected, never
silently degraded to a raw dict).
"""

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from llmfy.exception.llmfy_exception import CheckpointDeserializationException
from llmfy.flow_engine.checkpointer.serde import (
    TypeRegistry,
    deserialize_state,
    qualname,
    serialize_state,
)


class Point(BaseModel):
    x: int
    y: int


@dataclass
class Vector:
    dx: float
    dy: float


@dataclass
class Wrapper:
    label: str
    point: Point


class Unregistered(BaseModel):
    value: str


class TestTypeRegistry:
    def test_register_and_resolve_pydantic_model(self):
        registry = TypeRegistry()
        registry.register(Point)
        assert registry.resolve(qualname(Point)) is Point

    def test_register_and_resolve_dataclass(self):
        registry = TypeRegistry()
        registry.register(Vector)
        assert registry.resolve(qualname(Vector)) is Vector

    def test_constructor_accepts_types_list(self):
        registry = TypeRegistry(types=[Point, Vector])
        assert registry.resolve(qualname(Point)) is Point
        assert registry.resolve(qualname(Vector)) is Vector

    def test_resolve_unknown_returns_none(self):
        registry = TypeRegistry()
        assert registry.resolve("no.such.Type") is None

    def test_register_rejects_plain_class(self):
        class NotAllowed:
            pass

        registry = TypeRegistry()
        with pytest.raises(CheckpointDeserializationException):
            registry.register(NotAllowed)


class TestSerializeState:
    def test_json_native_values_pass_through(self):
        registry = TypeRegistry()
        state = {"a": 1, "b": "s", "c": 1.5, "d": True, "e": None, "f": [1, 2], "g": {"x": 1}}
        assert serialize_state(state, registry) == state

    def test_registered_pydantic_model_is_tagged(self):
        registry = TypeRegistry(types=[Point])
        out = serialize_state({"pos": Point(x=1, y=2)}, registry)
        assert out["pos"]["__type__"] == qualname(Point)
        assert out["pos"]["data"] == {"x": 1, "y": 2}

    def test_registered_dataclass_is_tagged(self):
        registry = TypeRegistry(types=[Vector])
        out = serialize_state({"v": Vector(dx=1.0, dy=2.0)}, registry)
        assert out["v"]["__type__"] == qualname(Vector)
        assert out["v"]["data"] == {"dx": 1.0, "dy": 2.0}

    def test_nested_registered_types_inside_dataclass_are_tagged(self):
        registry = TypeRegistry(types=[Wrapper, Point])
        out = serialize_state({"w": Wrapper(label="p1", point=Point(x=3, y=4))}, registry)
        assert out["w"]["__type__"] == qualname(Wrapper)
        assert out["w"]["data"]["point"]["__type__"] == qualname(Point)
        assert out["w"]["data"]["point"]["data"] == {"x": 3, "y": 4}

    def test_registered_type_inside_list_is_tagged(self):
        registry = TypeRegistry(types=[Point])
        out = serialize_state({"points": [Point(x=1, y=1), Point(x=2, y=2)]}, registry)
        assert out["points"][0]["__type__"] == qualname(Point)
        assert out["points"][1]["data"] == {"x": 2, "y": 2}

    def test_unregistered_type_raises_on_save(self):
        registry = TypeRegistry()
        with pytest.raises(CheckpointDeserializationException) as exc_info:
            serialize_state({"pos": Point(x=1, y=2)}, registry)
        assert exc_info.value.type_name == qualname(Point)
        assert exc_info.value.field_name == "pos"

    def test_unsupported_type_raises_on_save(self):
        registry = TypeRegistry()
        with pytest.raises(CheckpointDeserializationException):
            serialize_state({"obj": object()}, registry)


class TestDeserializeState:
    def test_round_trip_pydantic_model(self):
        registry = TypeRegistry(types=[Point])
        raw = serialize_state({"pos": Point(x=5, y=6)}, registry)
        restored = deserialize_state(raw, registry)
        assert restored["pos"] == Point(x=5, y=6)

    def test_round_trip_dataclass(self):
        registry = TypeRegistry(types=[Vector])
        raw = serialize_state({"v": Vector(dx=3.0, dy=4.0)}, registry)
        restored = deserialize_state(raw, registry)
        assert restored["v"] == Vector(dx=3.0, dy=4.0)

    def test_round_trip_nested_dataclass(self):
        registry = TypeRegistry(types=[Wrapper, Point])
        original = Wrapper(label="p1", point=Point(x=7, y=8))
        raw = serialize_state({"w": original}, registry)
        restored = deserialize_state(raw, registry)
        assert restored["w"] == original

    def test_round_trip_list_of_registered_type(self):
        registry = TypeRegistry(types=[Point])
        original = [Point(x=1, y=1), Point(x=2, y=2)]
        raw = serialize_state({"points": original}, registry)
        restored = deserialize_state(raw, registry)
        assert restored["points"] == original

    def test_unregistered_type_raises_on_load(self):
        registry_write = TypeRegistry(types=[Point])
        raw = serialize_state({"pos": Point(x=1, y=2)}, registry_write)

        registry_read = TypeRegistry()  # fresh process without Point registered
        with pytest.raises(CheckpointDeserializationException) as exc_info:
            deserialize_state(raw, registry_read)
        assert exc_info.value.type_name == qualname(Point)
        assert exc_info.value.field_name == "pos"

    def test_tampered_type_tag_raises_without_importing_anything(self):
        registry = TypeRegistry(types=[Point])
        tampered = {
            "pos": {"__type__": "os.system", "data": {"command": "rm -rf /"}}
        }
        with pytest.raises(CheckpointDeserializationException) as exc_info:
            deserialize_state(tampered, registry)
        assert exc_info.value.type_name == "os.system"

    def test_plain_dict_without_type_tag_passes_through(self):
        registry = TypeRegistry()
        state = {"meta": {"count": 3, "tags": ["a", "b"]}}
        assert deserialize_state(state, registry) == state
