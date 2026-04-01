from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from utils.dates import utc_now
from utils.ids import make_id


class AuditEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: make_id("audit"))
    timestamp: str = Field(default_factory=lambda: utc_now().isoformat())
    actor: str
    action: str
    target: str
    metadata: dict[str, Any] = Field(default_factory=dict)

