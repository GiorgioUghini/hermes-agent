"""Bounded, sequenced control-event delivery for realtime clients."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Mapping
import uuid

from gateway.realtime.protocol import control_event


@dataclass(frozen=True)
class ControlSubscription:
    queue: asyncio.Queue
    backlog: tuple[dict[str, Any], ...]
    cursor_expired: bool


class ControlEventBroker:
    """Single-loop broker with replay cursors and bounded subscriber queues."""

    def __init__(
        self,
        session_id: str,
        *,
        buffer_events: int = 256,
        subscriber_queue_events: int = 128,
    ):
        self.session_id = session_id
        self.stream_id = uuid.uuid4().hex
        self._events: Deque[dict[str, Any]] = deque(maxlen=max(1, buffer_events))
        self._queue_size = max(1, subscriber_queue_events)
        self._subscribers: set[asyncio.Queue] = set()
        self._sequence = 0

    @property
    def latest_sequence(self) -> int:
        return self._sequence

    def publish(
        self,
        event_type: str,
        data: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._sequence += 1
        event = control_event(
            sequence=self._sequence,
            session_id=self.session_id,
            stream_id=self.stream_id,
            event_type=event_type,
            data=data,
        )
        self._events.append(event)
        for queue in tuple(self._subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass
        return event

    def subscribe(self, *, after_sequence: int = 0) -> ControlSubscription:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.add(queue)
        oldest = self._events[0]["sequence"] if self._events else self._sequence + 1
        cursor_expired = bool(after_sequence and after_sequence < oldest - 1)
        backlog = tuple(
            event for event in self._events if event["sequence"] > after_sequence
        )
        return ControlSubscription(
            queue=queue,
            backlog=backlog,
            cursor_expired=cursor_expired,
        )

    def unsubscribe(self, subscription: ControlSubscription) -> None:
        self._subscribers.discard(subscription.queue)

    def snapshot(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._events)
