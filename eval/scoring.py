"""Deterministic scoring helpers (no LLM). Kept separate so they can be unit-tested."""
from __future__ import annotations

import re
from typing import Iterable, List

ABSTAIN_PHRASES = (
    "không tìm thấy", "không có thông tin", "không đề cập", "không được đề cập",
    "không báo cáo", "không nêu", "không cung cấp", "không có dữ liệu",
    "not found", "does not mention", "not mentioned", "not reported",
    "do not have enough information", "no information",
)


def normalize(text: str) -> str:
    t = (text or "").lower()
    t = t.replace("−", "-").replace("–", "-").replace("—", "-")   # minus / dashes
    t = t.replace(" ", " ").replace(" ", " ").replace(" ", " ")   # odd spaces
    t = re.sub(r"\*\*|__|`", "", t)                                               # markdown emphasis
    t = re.sub(r"(\d)\s*-\s*(\d)", r"\1-\2", t)                                   # "2 - 4" -> "2-4"
    t = re.sub(r"(\d)\s+%", r"\1%", t)                                            # "2 %" -> "2%"
    return t


def fact_hits(answer: str, key_facts: List[List[str]]) -> List[bool]:
    a = normalize(answer)
    return [any(normalize(alt) in a for alt in group) for group in key_facts]


def evidence_hit(retrieved_docs: Iterable[str], evidence: List[str]) -> bool:
    blob = "\n".join(retrieved_docs or [])
    return any(e in blob for e in evidence)


def abstained(answer: str) -> bool:
    a = normalize(answer)
    return any(p in a for p in ABSTAIN_PHRASES)
