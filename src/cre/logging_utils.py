"""Structured (JSON-lines) logging, so scheduled jobs produce machine-parseable logs in CI."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup(level: int = logging.INFO) -> None:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(level)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # per-request lines drown the signal


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields) -> None:
    """Log `msg` with structured key/value fields (shown as top-level JSON keys)."""
    logger.log(level, msg, extra={"fields": fields})
