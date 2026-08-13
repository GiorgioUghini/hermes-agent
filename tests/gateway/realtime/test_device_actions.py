from copy import deepcopy

import pytest

from gateway.realtime.device_actions import (
    DEVICE_CAPABILITY_CATALOG,
    DeviceCapabilityCatalog,
    DeviceCatalogError,
    DevicePayloadError,
)


DEVICE_ID = "22222222-2222-4222-8222-222222222222"


def _catalog_mapping():
    return {
        "protocol_version": 1,
        "catalog_version": 1,
        "capabilities": {
            "sample": {
                "versions": {
                    "1": {
                        "tools": [
                            {
                                "name": "device_sample_run",
                                "description": "Run the sample device action.",
                                "operation": "run",
                                "side_effecting": False,
                                "idempotency": "safe_to_retry",
                                "timeout_seconds": 10,
                                "parameters": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "value": {
                                            "type": "string",
                                            "maxLength": 20,
                                        }
                                    },
                                },
                                "result": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {},
                                },
                            }
                        ]
                    }
                }
            }
        },
    }


def test_timer_catalog_generates_trusted_model_tools_and_validates_payloads():
    binding = DEVICE_CAPABILITY_CATALOG.negotiate(
        protocol_version=1,
        instance_id=DEVICE_ID,
        capabilities=[("timer", 1), ("unsupported_future", 9)],
    )

    tools = binding.model_tools()
    assert [tool["function"]["name"] for tool in tools] == [
        "device_timer_create",
        "device_timer_list",
        "device_timer_cancel",
    ]
    create = binding.tool("device_timer_create")
    assert create.normalize_arguments({"duration_seconds": 300}) == {
        "duration_seconds": 300,
        "label": None,
    }
    assert create.validate_result({
        "timer_id": "timer-42",
        "fires_at_epoch_ms": 1_786_450_195_000,
        "label": None,
    })["timer_id"] == "timer-42"
    assert binding.public_payload()["capabilities"] == [
        {"name": "timer", "version": 1}
    ]

    with pytest.raises(DevicePayloadError):
        create.normalize_arguments({"duration_seconds": 0})
    with pytest.raises(DevicePayloadError):
        create.normalize_arguments({
            "duration_seconds": 30,
            "client_defined": True,
        })


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "unsafe_schema", "unsafe_timeout", "invalid_enum"],
)
def test_catalog_validation_rejects_unsafe_or_ambiguous_definitions(mutation):
    raw = _catalog_mapping()
    tool = raw["capabilities"]["sample"]["versions"]["1"]["tools"][0]
    if mutation == "duplicate":
        raw["capabilities"]["sample"]["versions"]["1"]["tools"].append(
            deepcopy(tool)
        )
    elif mutation == "unsafe_schema":
        tool["parameters"]["properties"]["value"]["maxLength"] = 100_000
    elif mutation == "unsafe_timeout":
        tool["timeout_seconds"] = 121
    else:
        tool["parameters"]["properties"]["value"]["enum"] = [42]

    with pytest.raises(DeviceCatalogError):
        DeviceCapabilityCatalog.from_mapping(raw)


def test_persisted_binding_detects_published_version_changes():
    binding = DEVICE_CAPABILITY_CATALOG.negotiate(
        protocol_version=1,
        instance_id=DEVICE_ID,
        capabilities=[("timer", 1)],
    )
    persisted = binding.persisted_payload()
    persisted["capabilities"][0]["fingerprint"] = "0" * 64

    with pytest.raises(DeviceCatalogError, match="changed incompatibly"):
        DEVICE_CAPABILITY_CATALOG.binding_from_persisted(persisted)
