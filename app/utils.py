from __future__ import annotations

import math
import re
import uuid
from typing import Iterable


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    clean = normalize_text(text)
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        chunks.append(clean[start:end])
        if end >= len(clean):
            break
        start = max(0, end - overlap)
    return chunks


def make_id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4().hex}"


def summarize_lines(items: Iterable[str], limit: int = 3) -> str:
    values = [item for item in items if item]
    if not values:
        return "None"
    return "; ".join(values[:limit])
