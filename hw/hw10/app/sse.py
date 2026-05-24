"""
SSE (Server-Sent Events) форматер.

Стандартний формат однієї SSE-події:
    data: <json>\\n\\n

Два \\n у кінці — обов'язковий роздільник між подіями. Без них клієнт
не зрозуміє що подія завершилась і триматиме в буфері.

Тип повернення — bytes (а не str), бо StreamingResponse у FastAPI працює з байтами
ефективніше: жодного зайвого encode на event.
"""
from __future__ import annotations

import json
from typing import Any


def format_event(payload: dict[str, Any]) -> bytes:
    """data: {...}\\n\\n у байтах. ensure_ascii=False — кирилиця читається у curl."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


# Стандартні заголовки для SSE-відповіді.
#   text/event-stream         — щоб клієнти (browsers, curl -N) інтерпретували як stream
#   no-cache                  — щоб проксі/браузер не кешували
#   X-Accel-Buffering: no     — вимикає буферизацію на nginx / Fly proxy
#                               (без цього токени летять одним блоком наприкінці)
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}
