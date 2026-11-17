"""Regression tests for the 'Table 2' failure (numbers were retrieved but never reached the LLM)."""
from app.utils.evidence import (
    build_evidence_digest, extract_final_answer, forced_answer_from_evidence,
    format_reference_docs, split_related_blocks,
)

TABLE = (
    "[RELATED TABLE]\nImage ID: _page_5_Figure_1\n"
    "| | Ours-JFT (ViT-L/16) | BiT-L (ResNet152x4) |\n|---|---|---|\n"
    "| ImageNet | 87.76 ± 0.03 | 87.54 ± 0.02 |\n| TPUv3-core-days | 0.68k | 9.9k |"
)
SECTION = "We first compare our largest models to state-of-the-art CNNs. " * 60   # ~3,500 chars, like section 4.2
DOC = SECTION + "\n" + TABLE


def test_old_truncation_lost_the_numbers():
    assert "0.68k" not in DOC[:1000]          # what the synthesizer used to see


def test_reference_docs_keep_related_tables():
    out = format_reference_docs([DOC])
    assert "0.68k" in out and "9.9k" in out and "87.76" in out
    assert len(out) < len(DOC)                 # prose was still clipped


def test_split_related_blocks():
    main, related = split_related_blocks(DOC)
    assert main.startswith("We first compare") and related == [TABLE]


def test_dedupe_and_budget():
    docs = [DOC, DOC] + [f"doc {i} " * 300 for i in range(40)]
    out = format_reference_docs(docs, total_chars=6000)
    assert out.count("Document [1]") == 1
    assert "more documents omitted" in out
    assert len(out) < 9000


def test_forced_answer_uses_evidence_not_free_text():
    essay = "### Analysis of Vision Transformer\nViTs are great and efficient..."
    ans = forced_answer_from_evidence(essay, [DOC])
    assert "0.68k" in ans and "Raw evidence" in ans and "great and efficient" not in ans
    assert forced_answer_from_evidence(essay, []) == essay.strip()


def test_digest_for_verifier_contains_table():
    assert "0.68k" in build_evidence_digest([DOC])


def test_final_answer_keeps_all_lines():
    text = "Thought: I have the data.\nFinal Answer: ViT-L/16 uses 0.68k [1].\n- BiT-L uses 9.9k [1].\n- Accuracy 87.76 vs 87.54 [1]."
    ans = extract_final_answer(text)
    assert "9.9k" in ans and "87.54" in ans      # old regex returned only the first line


def test_final_answer_markdown_bold_and_missing():
    assert extract_final_answer("**Final Answer:** 42") == "42"
    assert extract_final_answer("Thought: hmm\nAction: retrieve_tool\nAction Input: x") is None
