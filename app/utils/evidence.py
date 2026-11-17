"""
Helpers that decide HOW MUCH retrieved evidence each LLM step gets to see.

Why this exists (found while debugging the ViT "Table 2" question):
retrieved chunks look like

    <section text ...>
    [RELATED TABLE]
    Image ID: _page_5_Figure_1
    | ... | TPUv3-core-days | 0.68k | ... | 9.9k |

The synthesizer used to cut every document at 1,000 characters, so the
[RELATED TABLE] block at the end (where the numbers are) never reached the
LLM. Here prose is truncated, but related table/figure blocks are kept.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Tuple

_RELATED_RE = re.compile(r"^\[RELATED [A-Z_]+\]", re.MULTILINE)


def split_related_blocks(doc: str) -> Tuple[str, List[str]]:
    """Split a retrieved document into (main text, [related table/figure/equation blocks])."""
    starts = [m.start() for m in _RELATED_RE.finditer(doc)]
    if not starts:
        return doc.strip(), []
    main = doc[: starts[0]].strip()
    blocks = [doc[s:e].strip() for s, e in zip(starts, starts[1:] + [len(doc)])]
    return main, [b for b in blocks if b]


def _dedupe(docs: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for d in docs:
        d = (d or "").strip()
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + " ..."


def format_reference_docs(
    docs: Iterable[str],
    per_doc_chars: int = 1200,
    related_chars: int = 2500,
    total_chars: int = 24000,
) -> str:
    """
    Render documents as "- Document [i] contents: ..." lines for the synthesizer.
    Prose is clipped to `per_doc_chars`; each related block keeps up to
    `related_chars`; rendering stops once `total_chars` is reached.
    """
    parts: List[str] = []
    used = 0
    for i, doc in enumerate(_dedupe(docs), start=1):
        main, related = split_related_blocks(doc)
        body = _clip(main, per_doc_chars)
        if related:
            body += "\n" + "\n".join(_clip(b, related_chars) for b in related)
        entry = f"- Document [{i}] contents: {body}"
        if used + len(entry) > total_chars and parts:
            parts.append(f"- ({len(_dedupe(docs)) - i + 1} more documents omitted to fit the context window)")
            break
        parts.append(entry)
        used += len(entry)
    return "\n".join(parts)


def build_evidence_digest(docs: Iterable[str], total_chars: int = 8000) -> str:
    """Smaller version for the verifier and for the research-agent fallback."""
    return format_reference_docs(docs, per_doc_chars=600, related_chars=2000, total_chars=total_chars)


def forced_answer_from_evidence(response_text: str, retrieved_docs: List[str]) -> str:
    """
    The local LLM sometimes ignores the ReAct format (no 'Action:' / 'Final Answer:').
    Previously its free-form text became the task result, even when it was a generic
    essay that ignored the retrieved data. If we already retrieved evidence, pass that
    evidence on instead, so the verifier and synthesizer can still use the real facts.
    """
    if not retrieved_docs:
        return response_text.strip()
    return (
        "NOTE: The research step did not finish with a properly formatted answer. "
        "Use ONLY the raw evidence below.\n\n"
        "Raw evidence:\n" + build_evidence_digest(retrieved_docs, total_chars=10000)
    )


_FINAL_ANSWER_RE = re.compile(r"\**\s*Final Answer\s*\**\s*:\s*\**\s*(.*)", re.DOTALL | re.IGNORECASE)


def extract_final_answer(response_text: str) -> str | None:
    """
    Return everything after 'Final Answer:' (all lines), or None.
    The old regex stopped at the first newline, so multi-line answers
    (lists, tables) were cut to their first line.
    """
    m = _FINAL_ANSWER_RE.search(response_text or "")
    if not m:
        return None
    answer = m.group(1).strip()
    return answer or None
