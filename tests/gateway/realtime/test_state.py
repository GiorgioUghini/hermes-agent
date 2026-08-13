import sqlite3

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    state = SessionDB(db_path=tmp_path / "state.db")
    state.create_session(session_id="voice-state", source="realtime_voice")
    yield state
    state.close()


def _save_state(db, **overrides):
    values = {
        "provider_call_id": "call_1",
        "provider_call_started_at": 123.5,
        "state": "ready",
        "model": "gpt-realtime-snapshot",
        "voice": "cedar",
        "frozen_instructions": "frozen prompt",
        "frozen_tools": [{"type": "function", "name": "skill_view"}],
        "frozen_base_tools": [{"type": "function", "name": "skill_view"}],
        "device_context": {
            "protocol_version": 1,
            "catalog_version": 1,
            "instance_id": "22222222-2222-4222-8222-222222222222",
            "capabilities": [],
        },
    }
    values.update(overrides)
    db.save_realtime_session_state("voice-state", **values)


def test_realtime_state_round_trips_and_deletes(db):
    _save_state(db)

    state = db.get_realtime_session_state("voice-state")
    assert state["provider_call_id"] == "call_1"
    assert state["provider_call_started_at"] == 123.5
    assert state["model"] == "gpt-realtime-snapshot"
    assert state["voice"] == "cedar"
    assert state["frozen_instructions"] == "frozen prompt"
    assert state["frozen_tools"] == [{"type": "function", "name": "skill_view"}]
    assert state["frozen_base_tools"] == [
        {"type": "function", "name": "skill_view"}
    ]
    assert state["device_context"]["protocol_version"] == 1
    assert state["review_state"] == "idle"

    db.delete_realtime_session_state("voice-state")
    assert db.get_realtime_session_state("voice-state") is None


def test_realtime_device_invocation_is_first_writer_wins_and_durable(db):
    request = {
        "invocation_id": "device-call-1",
        "tool_name": "device_timer_create",
        "expires_at_epoch_ms": 1_786_450_200_000,
    }
    assert db.create_realtime_device_invocation(
        "voice-state",
        invocation_id="device-call-1",
        tool_name="device_timer_create",
        side_effecting=True,
        request_data=request,
        expires_at_epoch_ms=request["expires_at_epoch_ms"],
    )
    assert not db.create_realtime_device_invocation(
        "voice-state",
        invocation_id="device-call-1",
        tool_name="device_timer_create",
        side_effecting=True,
        request_data=request,
        expires_at_epoch_ms=request["expires_at_epoch_ms"],
    )
    pending = db.get_realtime_device_invocation(
        "voice-state",
        "device-call-1",
    )
    assert pending["status"] == "pending"
    assert pending["dispatch_state"] == "claimed"
    assert pending["request_data"] == request
    assert db.mark_realtime_device_invocation_dispatch(
        "voice-state",
        "device-call-1",
        dispatched=False,
    )
    assert db.mark_realtime_device_invocation_dispatch(
        "voice-state",
        "device-call-1",
        dispatched=True,
    )
    assert db.get_realtime_device_invocation(
        "voice-state",
        "device-call-1",
    )["dispatch_state"] == "dispatched"

    output = '{"success":true,"result":{"timer_id":"timer-42"}}'
    assert db.record_realtime_device_invocation_result(
        "voice-state",
        "device-call-1",
        output=output,
        effect_disposition=None,
        result_source="device",
    )
    assert not db.record_realtime_device_invocation_result(
        "voice-state",
        "device-call-1",
        output='{"success":false}',
        effect_disposition="unknown",
        result_source="recovery",
    )
    captured = db.get_realtime_device_invocation(
        "voice-state",
        "device-call-1",
    )
    assert captured["status"] == "result_received"
    assert captured["output"] == output
    assert captured["result_source"] == "device"
    assert captured["continuation_required"] is True
    assert db.complete_realtime_device_invocation(
        "voice-state",
        "device-call-1",
    )
    assert (
        db.get_realtime_device_invocation(
            "voice-state",
            "device-call-1",
        )["status"]
        == "completed"
    )
    assert db.list_unfinished_realtime_device_invocations("voice-state")
    assert db.clear_realtime_device_continuations("voice-state") == 1
    completed = db.get_realtime_device_invocation(
        "voice-state",
        "device-call-1",
    )
    assert completed["continuation_required"] is False
    assert db.list_unfinished_realtime_device_invocations("voice-state") == []
    assert db.require_realtime_device_continuations(
        "voice-state",
        ["device-call-1"],
    ) == 1
    assert db.has_realtime_device_continuation("voice-state")


def test_review_claims_are_single_winner_and_newer_boundaries_coalesce(db):
    _save_state(db)
    first = db.append_message("voice-state", role="user", content="first")
    db.mark_realtime_review_due(
        "voice-state",
        boundary_message_id=first,
        review_memory=True,
        review_skills=False,
    )

    assert db.mark_realtime_review_running(
        "voice-state", boundary_message_id=first
    )
    assert not db.mark_realtime_review_running(
        "voice-state", boundary_message_id=first
    )

    second = db.append_message("voice-state", role="assistant", content="second")
    db.mark_realtime_review_due(
        "voice-state",
        boundary_message_id=second,
        review_memory=False,
        review_skills=True,
    )
    queued = db.get_realtime_session_state("voice-state")
    assert queued["review_state"] == "due"
    assert queued["review_boundary_message_id"] == second
    assert queued["review_memory"] is True
    assert queued["review_skills"] is True

    assert not db.finish_realtime_review(
        "voice-state",
        boundary_message_id=first,
        success=True,
    )
    assert db.mark_realtime_review_running(
        "voice-state", boundary_message_id=second
    )
    assert db.finish_realtime_review(
        "voice-state",
        boundary_message_id=second,
        success=True,
    )
    assert db.get_realtime_session_state("voice-state")["review_state"] == "completed"


def test_realtime_columns_reconcile_on_existing_database(tmp_path):
    path = tmp_path / "legacy-state.db"
    initial = SessionDB(db_path=path)
    initial.close()

    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE realtime_sessions")
        conn.execute(
            """CREATE TABLE realtime_sessions (
                session_id TEXT PRIMARY KEY,
                provider_call_id TEXT,
                provider_call_started_at REAL,
                state TEXT NOT NULL DEFAULT 'ready',
                frozen_tools TEXT NOT NULL DEFAULT '[]',
                review_state TEXT NOT NULL DEFAULT 'idle',
                review_boundary_message_id INTEGER,
                review_memory INTEGER NOT NULL DEFAULT 0,
                review_skills INTEGER NOT NULL DEFAULT 0,
                review_error TEXT,
                updated_at REAL NOT NULL
            )"""
        )

    reconciled = SessionDB(db_path=path)
    try:
        with sqlite3.connect(path) as conn:
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(realtime_sessions)")
            }
            indexes = {
                row[1] for row in conn.execute("PRAGMA index_list(realtime_sessions)")
            }
        assert "frozen_instructions" in columns
        assert {"model", "voice"} <= columns
        assert {"frozen_base_tools", "device_context"} <= columns
        assert "idx_realtime_sessions_review" in indexes
        device_columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(realtime_device_invocations)"
            )
        }
        assert {
            "session_id",
            "invocation_id",
            "status",
            "output",
            "expires_at_epoch_ms",
            "dispatch_state",
        } <= device_columns
    finally:
        reconciled.close()
