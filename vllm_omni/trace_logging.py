from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

_TRACE_LOCK = threading.Lock()


def write_trace_event(
    trace_file: str | None,
    event: str,
    *,
    node: str | None = None,
    request_id: str | None = None,
    **fields: Any,
) -> None:
    if not trace_file:
        return

    record: dict[str, Any] = {
        "ts": time.time(),
        "ts_ns": time.time_ns(),
        "pid": os.getpid(),
        "event": event,
    }
    if node:
        record["node"] = node
    if request_id:
        record["request_id"] = request_id
    record.update(fields)

    os.makedirs(os.path.dirname(trace_file) or ".", exist_ok=True)
    line = json.dumps(record, sort_keys=True, ensure_ascii=True)
    with _TRACE_LOCK:
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")
