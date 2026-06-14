# -*- coding: utf-8 -*-
"""Event bus for broadcasting server-side events (SSE).

All events are broadcast as BusEvent instances with the opencode-style format:
    {"id": "<event_id>", "type": "<event_type>", "properties": {...}}

The SSE wire format mirrors opencode exactly: each event is encoded as
    event: message\\n
    id: <event_id>\\n
    data: {"id":...,"type":...,"properties":...}\\n\\n
"""

from __future__ import annotations

import asyncio
import orjson
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Event ID generator (opencode-style ascending evt_ IDs) ──────────

_id_lock = threading.Lock()
_id_counter = 0


def create_id() -> str:
    """Generate an opencode-style ascending event ID (``evt_`` prefix)."""
    global _id_counter
    with _id_lock:
        _id_counter += 1
        counter = _id_counter
    ts_hex = format(int(time.time() * 1000), "012x")
    cnt_hex = format(counter, "06x")
    return f"evt_{ts_hex}{cnt_hex}"


@dataclass
class BusEvent:
    """A structured event to be broadcast via SSE.

    Format matches opencode protocol:
        {"id": "evt_...", "type": "<event_type>", "properties": {...}}
    """

    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=create_id)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "type": self.type, "properties": self.properties}

    def to_json(self) -> str:
        return orjson.dumps(self.to_dict()).decode("utf-8")

    def to_sse(self) -> str:
        """Encode as an opencode-compatible SSE frame.

        opencode uses a named ``message`` event plus an ``id`` line so the
        client can populate the ``Last-Event-ID`` header on reconnect.
        """
        return f"event: message\nid: {self.id}\ndata: {self.to_json()}\n\n"


class EventBus:
    """Simple pub/sub event bus supporting both sync and async subscribers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: List[Callable[[BusEvent], None]] = []
        self._async_queues: List[asyncio.Queue[Optional[BusEvent]]] = []

    def subscribe(self, callback: Callable[[BusEvent], None]) -> Callable[[], None]:
        """Subscribe with a sync callback. Returns an unsubscribe function."""
        with self._lock:
            self._subscribers.append(callback)

        def _unsub() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(callback)
                except ValueError:
                    pass

        return _unsub

    def create_async_queue(self) -> asyncio.Queue[Optional[BusEvent]]:
        """Create an asyncio queue that receives all events. Returns queue."""
        q: asyncio.Queue[Optional[BusEvent]] = asyncio.Queue()
        with self._lock:
            self._async_queues.append(q)
        return q

    def remove_async_queue(self, q: asyncio.Queue[Optional[BusEvent]]) -> None:
        with self._lock:
            try:
                self._async_queues.remove(q)
            except ValueError:
                pass

    def get_all_queues(self) -> List[asyncio.Queue[Optional[BusEvent]]]:
        with self._lock:
            return list(self._async_queues)

    def emit(self, event: BusEvent) -> None:
        """Emit an event to all subscribers."""
        with self._lock:
            subs = list(self._subscribers)
            queues = list(self._async_queues)

        for cb in subs:
            try:
                cb(event)
            except Exception:
                logger.debug("Event subscriber error", exc_info=True)

        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("SSE queue full, dropping event")
            except Exception:
                logger.debug("Queue put error", exc_info=True)

    def emit_type(self, event_type: str, **properties: Any) -> None:
        """Convenience: emit by type name and keyword properties."""
        self.emit(BusEvent(type=event_type, properties=properties))


# Global singleton
bus = EventBus()
