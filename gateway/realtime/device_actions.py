"""Trusted catalog and session-scoped bindings for device-executed tools."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional


_CAPABILITY_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_TOOL_NAME_RE = re.compile(r"^device_[a-z][a-z0-9_]{0,63}$")
_OPERATION_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SCHEMA_TYPES = frozenset({
    "array",
    "boolean",
    "integer",
    "null",
    "number",
    "object",
    "string",
})
_SCHEMA_KEYS = frozenset({
    "additionalProperties",
    "default",
    "description",
    "enum",
    "items",
    "maxItems",
    "maxLength",
    "maximum",
    "minItems",
    "minLength",
    "minimum",
    "properties",
    "required",
    "type",
})
_IDEMPOTENCY_VALUES = frozenset({"at_most_once", "safe_to_retry"})
_MAX_SCHEMA_DEPTH = 8
_MAX_SCHEMA_PROPERTIES = 64
_MAX_SCHEMA_STRING_LENGTH = 16 * 1024
_MAX_SCHEMA_ARRAY_ITEMS = 256
_MAX_CATALOG_TOOLS = 128
_MAX_TOOL_TIMEOUT_SECONDS = 120


class DeviceCatalogError(ValueError):
    """The checked-in device capability catalog is invalid."""


class DevicePayloadError(ValueError):
    """A model argument or device result violates a catalog schema."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DeviceCatalogError(f"Duplicate catalog key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_number(value):
    raise DeviceCatalogError(f"Catalog contains a non-finite number: {value}")


def _is_finite_number(value: int | float) -> bool:
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _schema_types(schema: Mapping[str, Any], *, path: str) -> tuple[str, ...]:
    raw = schema.get("type")
    if isinstance(raw, str):
        types = (raw,)
    elif isinstance(raw, list) and raw:
        types = tuple(raw)
    else:
        raise DeviceCatalogError(f"{path}.type must be a string or non-empty list")
    if (
        any(not isinstance(item, str) or item not in _SCHEMA_TYPES for item in types)
        or len(set(types)) != len(types)
        or len([item for item in types if item != "null"]) > 1
    ):
        raise DeviceCatalogError(f"{path}.type contains an unsupported union")
    return types


def _positive_bounded_int(
    value: Any,
    *,
    path: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DeviceCatalogError(f"{path} must be an integer")
    if not minimum <= value <= maximum:
        raise DeviceCatalogError(f"{path} must be between {minimum} and {maximum}")
    return value


def _validate_schema_definition(
    raw: Any,
    *,
    path: str,
    depth: int = 0,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise DeviceCatalogError(f"{path} must be an object")
    if depth > _MAX_SCHEMA_DEPTH:
        raise DeviceCatalogError(f"{path} exceeds the maximum schema depth")
    unknown = set(raw) - _SCHEMA_KEYS
    if unknown:
        raise DeviceCatalogError(
            f"{path} contains unsupported schema keys: {sorted(unknown)}"
        )
    schema = deepcopy(dict(raw))
    types = _schema_types(schema, path=path)
    primary_type = next((item for item in types if item != "null"), "null")

    if "description" in schema and (
        not isinstance(schema["description"], str)
        or not 1 <= len(schema["description"].strip()) <= 500
        or any(ch in schema["description"] for ch in ("\r", "\n", "\x00"))
    ):
        raise DeviceCatalogError(f"{path}.description is invalid")

    if "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, list) or not enum or len(enum) > 64:
            raise DeviceCatalogError(f"{path}.enum must contain 1 to 64 values")
        schema_without_enum = dict(schema)
        schema_without_enum.pop("enum")
        for index, item in enumerate(enum):
            try:
                validate_schema_value(
                    schema_without_enum,
                    item,
                    path=f"{path}.enum[{index}]",
                )
            except DevicePayloadError as exc:
                raise DeviceCatalogError(str(exc)) from exc
        if len(_canonical_json(enum)) > _MAX_SCHEMA_STRING_LENGTH:
            raise DeviceCatalogError(f"{path}.enum is too large")

    if primary_type == "object":
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            raise DeviceCatalogError(f"{path}.properties must be an object")
        if len(properties) > _MAX_SCHEMA_PROPERTIES:
            raise DeviceCatalogError(f"{path} has too many properties")
        if schema.get("additionalProperties") is not False:
            raise DeviceCatalogError(f"{path}.additionalProperties must be false")
        normalized_properties = {}
        for name, child in properties.items():
            if not isinstance(name, str) or not _CAPABILITY_NAME_RE.fullmatch(name):
                raise DeviceCatalogError(f"{path}.properties contains an invalid name")
            normalized_properties[name] = _validate_schema_definition(
                child,
                path=f"{path}.properties.{name}",
                depth=depth + 1,
            )
        schema["properties"] = normalized_properties
        required = schema.get("required", [])
        if (
            not isinstance(required, list)
            or any(not isinstance(item, str) for item in required)
            or len(set(required)) != len(required)
            or any(item not in normalized_properties for item in required)
        ):
            raise DeviceCatalogError(f"{path}.required is invalid")
    elif any(key in schema for key in ("properties", "required", "additionalProperties")):
        raise DeviceCatalogError(f"{path} uses object keywords on a non-object schema")

    if primary_type == "array":
        if "items" not in schema:
            raise DeviceCatalogError(f"{path}.items is required")
        schema["items"] = _validate_schema_definition(
            schema["items"],
            path=f"{path}.items",
            depth=depth + 1,
        )
        max_items = _positive_bounded_int(
            schema.get("maxItems"),
            path=f"{path}.maxItems",
            minimum=0,
            maximum=_MAX_SCHEMA_ARRAY_ITEMS,
        )
        min_items = schema.get("minItems", 0)
        if isinstance(min_items, bool) or not isinstance(min_items, int):
            raise DeviceCatalogError(f"{path}.minItems must be an integer")
        if not 0 <= min_items <= max_items:
            raise DeviceCatalogError(f"{path}.minItems is invalid")
    elif any(key in schema for key in ("items", "minItems", "maxItems")):
        raise DeviceCatalogError(f"{path} uses array keywords on a non-array schema")

    if primary_type == "string":
        max_length = _positive_bounded_int(
            schema.get("maxLength"),
            path=f"{path}.maxLength",
            minimum=1,
            maximum=_MAX_SCHEMA_STRING_LENGTH,
        )
        min_length = schema.get("minLength", 0)
        if isinstance(min_length, bool) or not isinstance(min_length, int):
            raise DeviceCatalogError(f"{path}.minLength must be an integer")
        if not 0 <= min_length <= max_length:
            raise DeviceCatalogError(f"{path}.minLength is invalid")
    elif any(key in schema for key in ("minLength", "maxLength")):
        raise DeviceCatalogError(f"{path} uses string keywords on a non-string schema")

    if primary_type in {"integer", "number"}:
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if (
            isinstance(minimum, bool)
            or isinstance(maximum, bool)
            or not isinstance(minimum, (int, float))
            or not isinstance(maximum, (int, float))
            or not _is_finite_number(minimum)
            or not _is_finite_number(maximum)
            or minimum > maximum
            or abs(minimum) > 10**16
            or abs(maximum) > 10**16
        ):
            raise DeviceCatalogError(f"{path} requires safe minimum and maximum bounds")
    elif any(key in schema for key in ("minimum", "maximum")):
        raise DeviceCatalogError(f"{path} uses numeric keywords on a non-number schema")

    if "default" in schema:
        try:
            validate_schema_value(schema, schema["default"], path=f"{path}.default")
        except DevicePayloadError as exc:
            raise DeviceCatalogError(str(exc)) from exc
    return schema


def validate_schema_value(
    schema: Mapping[str, Any],
    value: Any,
    *,
    path: str = "$",
) -> None:
    """Validate a value against the deliberately small catalog schema subset."""

    raw_type = schema.get("type")
    types = (raw_type,) if isinstance(raw_type, str) else tuple(raw_type or ())
    if value is None:
        if "null" in types:
            return
        raise DevicePayloadError(f"{path} must not be null")
    primary_type = next((item for item in types if item != "null"), "null")

    valid_type = {
        "array": lambda item: isinstance(item, list),
        "boolean": lambda item: isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float))
        and not isinstance(item, bool),
        "object": lambda item: isinstance(item, Mapping),
        "string": lambda item: isinstance(item, str),
    }.get(primary_type)
    if valid_type is None or not valid_type(value):
        raise DevicePayloadError(f"{path} must be {primary_type}")
    if "enum" in schema and value not in schema["enum"]:
        raise DevicePayloadError(f"{path} is not an allowed value")

    if primary_type == "object":
        properties = schema["properties"]
        unknown = set(value) - set(properties)
        if unknown:
            raise DevicePayloadError(f"{path} contains unknown fields: {sorted(unknown)}")
        missing = set(schema.get("required", [])) - set(value)
        if missing:
            raise DevicePayloadError(f"{path} is missing required fields: {sorted(missing)}")
        for name, child in properties.items():
            if name in value:
                validate_schema_value(child, value[name], path=f"{path}.{name}")
    elif primary_type == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema["maxItems"]:
            raise DevicePayloadError(f"{path} has an invalid item count")
        for index, item in enumerate(value):
            validate_schema_value(schema["items"], item, path=f"{path}[{index}]")
    elif primary_type == "string":
        if not schema.get("minLength", 0) <= len(value) <= schema["maxLength"]:
            raise DevicePayloadError(f"{path} has an invalid length")
    elif primary_type in {"integer", "number"}:
        if not schema["minimum"] <= value <= schema["maximum"]:
            raise DevicePayloadError(f"{path} is outside the allowed range")


def _with_schema_defaults(
    schema: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = deepcopy(dict(value))
    for name, child in schema.get("properties", {}).items():
        if name not in normalized and "default" in child:
            normalized[name] = deepcopy(child["default"])
    return normalized


@dataclass(frozen=True)
class DeviceToolDefinition:
    name: str
    description: str
    capability: str
    capability_version: int
    operation: str
    side_effecting: bool
    idempotency: str
    timeout_seconds: int
    parameters: dict[str, Any]
    result_schema: dict[str, Any]

    def model_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": deepcopy(self.parameters),
            },
        }

    def normalize_arguments(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise DevicePayloadError("arguments must be an object")
        normalized = _with_schema_defaults(self.parameters, value)
        validate_schema_value(self.parameters, normalized, path="arguments")
        return normalized

    def validate_result(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise DevicePayloadError("result must be an object")
        normalized = deepcopy(dict(value))
        validate_schema_value(self.result_schema, normalized, path="result")
        return normalized


@dataclass(frozen=True)
class DeviceSessionBinding:
    protocol_version: int
    catalog_version: int
    instance_id: str
    capabilities: tuple[tuple[str, int, str], ...]
    tools: tuple[DeviceToolDefinition, ...]

    def public_payload(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "instance_id": self.instance_id,
            "capabilities": [
                {"name": name, "version": version}
                for name, version, _fingerprint in self.capabilities
            ],
        }

    def persisted_payload(self) -> dict[str, Any]:
        return {
            **self.public_payload(),
            "catalog_version": self.catalog_version,
            "capabilities": [
                {
                    "name": name,
                    "version": version,
                    "fingerprint": fingerprint,
                }
                for name, version, fingerprint in self.capabilities
            ],
        }

    def model_tools(self) -> list[dict[str, Any]]:
        return [tool.model_tool() for tool in self.tools]

    def tool(self, name: str) -> Optional[DeviceToolDefinition]:
        return next((tool for tool in self.tools if tool.name == name), None)


class DeviceCapabilityCatalog:
    """Validated immutable capability versions and their trusted tool schemas."""

    def __init__(self, raw: Mapping[str, Any]):
        if not isinstance(raw, Mapping):
            raise DeviceCatalogError("Device capability catalog must be an object")
        unknown = set(raw) - {"protocol_version", "catalog_version", "capabilities"}
        if unknown:
            raise DeviceCatalogError(f"Unknown catalog fields: {sorted(unknown)}")
        self.protocol_version = _positive_bounded_int(
            raw.get("protocol_version"),
            path="protocol_version",
            minimum=1,
            maximum=1000,
        )
        self.catalog_version = _positive_bounded_int(
            raw.get("catalog_version"),
            path="catalog_version",
            minimum=1,
            maximum=1_000_000,
        )
        capabilities = raw.get("capabilities")
        if not isinstance(capabilities, Mapping) or not capabilities:
            raise DeviceCatalogError("capabilities must be a non-empty object")

        self._versions: dict[tuple[str, int], tuple[DeviceToolDefinition, ...]] = {}
        self._fingerprints: dict[tuple[str, int], str] = {}
        tool_capabilities: dict[str, str] = {}
        total_tools = 0
        for capability, capability_raw in capabilities.items():
            if (
                not isinstance(capability, str)
                or not _CAPABILITY_NAME_RE.fullmatch(capability)
                or not isinstance(capability_raw, Mapping)
                or set(capability_raw) != {"versions"}
            ):
                raise DeviceCatalogError(f"Invalid capability entry: {capability!r}")
            versions = capability_raw["versions"]
            if not isinstance(versions, Mapping) or not versions:
                raise DeviceCatalogError(f"{capability}.versions must be non-empty")
            for raw_version, version_raw in versions.items():
                try:
                    version = int(raw_version)
                except (TypeError, ValueError) as exc:
                    raise DeviceCatalogError(
                        f"{capability} has an invalid version"
                    ) from exc
                if str(version) != str(raw_version) or not 1 <= version <= 1_000_000:
                    raise DeviceCatalogError(f"{capability} has an invalid version")
                tools = self._validate_version(capability, version, version_raw)
                total_tools += len(tools)
                if total_tools > _MAX_CATALOG_TOOLS:
                    raise DeviceCatalogError("Catalog contains too many tools")
                for tool in tools:
                    prior_capability = tool_capabilities.setdefault(
                        tool.name, capability
                    )
                    if prior_capability != capability:
                        raise DeviceCatalogError(
                            f"Tool {tool.name} is declared by multiple capabilities"
                        )
                key = (capability, version)
                self._versions[key] = tools
                self._fingerprints[key] = hashlib.sha256(
                    _canonical_json({
                        "capability": capability,
                        "version": version,
                        "definition": version_raw,
                    }).encode("ascii")
                ).hexdigest()

        self._tools_by_name = {
            tool.name: tool
            for tools in self._versions.values()
            for tool in tools
        }
        self._all_tool_names = frozenset(self._tools_by_name)

    @classmethod
    def from_path(cls, path: Path) -> "DeviceCapabilityCatalog":
        try:
            raw = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite_number,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DeviceCatalogError(f"Could not load device catalog: {exc}") from exc
        return cls(raw)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DeviceCapabilityCatalog":
        return cls(deepcopy(dict(raw)))

    def _validate_version(
        self,
        capability: str,
        version: int,
        raw: Any,
    ) -> tuple[DeviceToolDefinition, ...]:
        if not isinstance(raw, Mapping) or set(raw) != {"tools"}:
            raise DeviceCatalogError(f"{capability}.versions.{version} is invalid")
        raw_tools = raw["tools"]
        if not isinstance(raw_tools, list) or not raw_tools:
            raise DeviceCatalogError(
                f"{capability}.versions.{version}.tools must be non-empty"
            )
        tools = []
        names = set()
        for index, raw_tool in enumerate(raw_tools):
            path = f"{capability}.versions.{version}.tools[{index}]"
            expected = {
                "description",
                "idempotency",
                "name",
                "operation",
                "parameters",
                "result",
                "side_effecting",
                "timeout_seconds",
            }
            if not isinstance(raw_tool, Mapping) or set(raw_tool) != expected:
                raise DeviceCatalogError(f"{path} has missing or unknown fields")
            name = raw_tool["name"]
            description = raw_tool["description"]
            operation = raw_tool["operation"]
            idempotency = raw_tool["idempotency"]
            if not isinstance(name, str) or not _TOOL_NAME_RE.fullmatch(name):
                raise DeviceCatalogError(f"{path}.name is invalid")
            if name in names:
                raise DeviceCatalogError(f"Duplicate tool name in {capability} v{version}")
            names.add(name)
            if (
                not isinstance(description, str)
                or not 1 <= len(description.strip()) <= 500
                or any(ch in description for ch in ("\r", "\n", "\x00"))
            ):
                raise DeviceCatalogError(f"{path}.description is invalid")
            if not isinstance(operation, str) or not _OPERATION_RE.fullmatch(operation):
                raise DeviceCatalogError(f"{path}.operation is invalid")
            if not isinstance(raw_tool["side_effecting"], bool):
                raise DeviceCatalogError(f"{path}.side_effecting must be boolean")
            if idempotency not in _IDEMPOTENCY_VALUES:
                raise DeviceCatalogError(f"{path}.idempotency is invalid")
            if raw_tool["side_effecting"] and idempotency == "safe_to_retry":
                raise DeviceCatalogError(
                    f"{path} cannot mark a side-effecting tool safe_to_retry"
                )
            timeout = _positive_bounded_int(
                raw_tool["timeout_seconds"],
                path=f"{path}.timeout_seconds",
                minimum=1,
                maximum=_MAX_TOOL_TIMEOUT_SECONDS,
            )
            parameters = _validate_schema_definition(
                raw_tool["parameters"],
                path=f"{path}.parameters",
            )
            result_schema = _validate_schema_definition(
                raw_tool["result"],
                path=f"{path}.result",
            )
            if parameters.get("type") != "object" or result_schema.get("type") != "object":
                raise DeviceCatalogError(f"{path} parameters and result must be objects")
            tools.append(
                DeviceToolDefinition(
                    name=name,
                    description=description.strip(),
                    capability=capability,
                    capability_version=version,
                    operation=operation,
                    side_effecting=raw_tool["side_effecting"],
                    idempotency=idempotency,
                    timeout_seconds=timeout,
                    parameters=parameters,
                    result_schema=result_schema,
                )
            )
        return tuple(tools)

    def advertised_capabilities(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for capability, version in self._versions:
            result[capability] = max(version, result.get(capability, 0))
        return dict(sorted(result.items()))

    def is_device_tool(self, name: str) -> bool:
        return name in self._all_tool_names

    def tool(self, name: str) -> Optional[DeviceToolDefinition]:
        return self._tools_by_name.get(name)

    def negotiate(
        self,
        *,
        protocol_version: int,
        instance_id: str,
        capabilities: Iterable[tuple[str, int]],
    ) -> DeviceSessionBinding:
        if protocol_version != self.protocol_version:
            raise DeviceCatalogError(
                f"Unsupported device protocol version: {protocol_version}"
            )
        accepted = []
        tools = []
        seen_capabilities = set()
        seen_tool_names = set()
        for capability, version in capabilities:
            if capability in seen_capabilities:
                raise DeviceCatalogError(
                    f"Capability {capability} advertises more than one version"
                )
            seen_capabilities.add(capability)
            key = (capability, version)
            resolved = self._versions.get(key)
            if resolved is None:
                continue
            for tool in resolved:
                if tool.name in seen_tool_names:
                    raise DeviceCatalogError(
                        f"Negotiated capabilities duplicate tool {tool.name}"
                    )
                seen_tool_names.add(tool.name)
                tools.append(tool)
            accepted.append((capability, version, self._fingerprints[key]))
        return DeviceSessionBinding(
            protocol_version=protocol_version,
            catalog_version=self.catalog_version,
            instance_id=instance_id,
            capabilities=tuple(accepted),
            tools=tuple(tools),
        )

    def binding_from_persisted(
        self,
        raw: Mapping[str, Any] | None,
    ) -> Optional[DeviceSessionBinding]:
        if not raw:
            return None
        if not isinstance(raw, Mapping):
            raise DeviceCatalogError("Persisted device context is invalid")
        protocol_version = raw.get("protocol_version")
        catalog_version = raw.get("catalog_version")
        instance_id = raw.get("instance_id")
        capabilities = raw.get("capabilities")
        if (
            protocol_version != self.protocol_version
            or isinstance(catalog_version, bool)
            or not isinstance(catalog_version, int)
            or not isinstance(instance_id, str)
            or not isinstance(capabilities, list)
        ):
            raise DeviceCatalogError("Persisted device context is invalid")
        requested = []
        expected_fingerprints = {}
        for item in capabilities:
            if not isinstance(item, Mapping):
                raise DeviceCatalogError("Persisted capability is invalid")
            name = item.get("name")
            version = item.get("version")
            fingerprint = item.get("fingerprint")
            if (
                not isinstance(name, str)
                or isinstance(version, bool)
                or not isinstance(version, int)
                or not isinstance(fingerprint, str)
            ):
                raise DeviceCatalogError("Persisted capability is invalid")
            requested.append((name, version))
            expected_fingerprints[(name, version)] = fingerprint
        binding = self.negotiate(
            protocol_version=protocol_version,
            instance_id=instance_id,
            capabilities=requested,
        )
        if len(binding.capabilities) != len(requested):
            raise DeviceCatalogError("A persisted capability version is unavailable")
        for name, version, fingerprint in binding.capabilities:
            if expected_fingerprints.get((name, version)) != fingerprint:
                raise DeviceCatalogError(
                    f"Published capability {name} v{version} changed incompatibly"
                )
        return DeviceSessionBinding(
            protocol_version=binding.protocol_version,
            catalog_version=catalog_version,
            instance_id=binding.instance_id,
            capabilities=binding.capabilities,
            tools=binding.tools,
        )

    def strip_device_tools(
        self,
        tools: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        result = []
        for raw_tool in tools:
            if not isinstance(raw_tool, Mapping):
                continue
            function = raw_tool.get("function")
            function = function if isinstance(function, Mapping) else raw_tool
            if function.get("name") not in self._all_tool_names:
                result.append(deepcopy(dict(raw_tool)))
        return result


DEVICE_CAPABILITY_CATALOG = DeviceCapabilityCatalog.from_path(
    Path(__file__).with_name("device_capabilities.json")
)
