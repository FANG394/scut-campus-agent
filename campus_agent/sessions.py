from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass, field

from campus_agent.models import ChatMessage


_VALID_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{8,80}$")


@dataclass(slots=True)
class _Session:
    messages: list[ChatMessage] = field(default_factory=list)
    touched_at: float = field(default_factory=time.time)
    turn_count: int = 0
    last_tool_name: str | None = None
    last_tool_turn: int | None = None


class SessionStore:
    def __init__(
        self,
        *,
        max_messages: int = 12,
        ttl_seconds: int = 6 * 60 * 60,
        max_sessions: int = 500,
    ) -> None:
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()

    def normalise_id(self, requested: str | None) -> str:
        if requested and _VALID_SESSION_ID.fullmatch(requested):
            return requested
        return uuid.uuid4().hex

    def history(self, session_id: str) -> list[ChatMessage]:
        with self._lock:
            self._prune()
            session = self._sessions.get(session_id)
            if not session:
                return []
            session.touched_at = time.time()
            return [ChatMessage(role=item.role, content=item.content) for item in session.messages]

    def recent_tool_name(
        self, session_id: str, *, max_intervening_turns: int = 2
    ) -> str | None:
        """Return a recent tool domain without retaining any tool arguments."""

        if max_intervening_turns < 0:
            raise ValueError("max_intervening_turns 不能为负数。")
        with self._lock:
            self._prune()
            session = self._sessions.get(session_id)
            if (
                not session
                or not session.last_tool_name
                or session.last_tool_turn is None
            ):
                return None
            if session.turn_count - session.last_tool_turn > max_intervening_turns:
                session.last_tool_name = None
                session.last_tool_turn = None
                return None
            session.touched_at = time.time()
            return session.last_tool_name

    def append_exchange(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        *,
        tool_name: str | None = None,
    ) -> None:
        with self._lock:
            self._prune()
            session = self._sessions.setdefault(session_id, _Session())
            session.messages.extend(
                [
                    ChatMessage(role="user", content=user_text),
                    ChatMessage(role="assistant", content=assistant_text),
                ]
            )
            session.messages = session.messages[-self.max_messages :]
            session.turn_count += 1
            if tool_name:
                session.last_tool_name = tool_name
                session.last_tool_turn = session.turn_count
            session.touched_at = time.time()
            if len(self._sessions) > self.max_sessions:
                oldest = min(self._sessions, key=lambda key: self._sessions[key].touched_at)
                if oldest != session_id:
                    self._sessions.pop(oldest, None)

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _prune(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        expired = [key for key, session in self._sessions.items() if session.touched_at < cutoff]
        for key in expired:
            self._sessions.pop(key, None)
