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
    assert state["review_state"] == "idle"

    db.delete_realtime_session_state("voice-state")
    assert db.get_realtime_session_state("voice-state") is None


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
        assert "idx_realtime_sessions_review" in indexes
    finally:
        reconciled.close()
