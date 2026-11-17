import os

from app.services.chunker import SmartChunker, process_chunks

VIT_MD = os.path.join(os.path.dirname(__file__), "..", "storage", "processed",
                      "90034a5d-b97f-4a34-bacc-ef11780f5c89", "markdown", "result.md")


def test_long_section_is_split_on_paragraphs():
    c = SmartChunker(max_chunk_chars=500)
    paras = [("para %d " % i) * 20 for i in range(6)]   # ~140 chars each
    parts = c.split_long_chunk({"content": "\n\n".join(paras), "heading_path": "A > B", "section_title": "B"})
    assert len(parts) > 1
    assert all(len(p["content"]) <= 500 for p in parts)
    assert "\n\n".join(p["content"] for p in parts) == "\n\n".join(paras)   # nothing lost
    assert parts[0]["section_title"] == "B (part 1/%d)" % len(parts)


def test_short_section_untouched():
    c = SmartChunker()
    ch = {"content": "short", "heading_path": "A", "section_title": "A"}
    assert c.split_long_chunk(ch) == [ch]


def test_table_chunks_from_images(tmp_path):
    img = tmp_path / "images" / "_page_5_Figure_1"
    img.mkdir(parents=True)
    (img / "content.md").write_text("| a | b |\n|---|---|\n| TPUv3-core-days | 0.68k |", encoding="utf-8")
    eq = tmp_path / "images" / "_page_3_Figure_2"
    eq.mkdir(parents=True)
    (eq / "content.md").write_text("$$ z_0 = x $$", encoding="utf-8")
    md = ("## 4 EXPERIMENTS\n#### 4.2 COMPARISON\n\n![](../images/_page_5_Figure_1/image.png)\n\n"
          "<span id=\"page-5-0\"></span>Table 2: Comparison with state of the art.\n\n"
          "![](../images/_page_3_Figure_2/image.png)\n")
    chunks = SmartChunker().build_table_chunks(md, str(tmp_path / "images"))
    assert len(chunks) == 1                                    # the equation is not a table
    t = chunks[0]
    assert t["section_title"] == "Table 2"
    assert t["content"].startswith("Table 2: Comparison with state of the art.")
    assert "0.68k" in t["content"]
    assert t["heading_path"].endswith("4.2 COMPARISON > Table 2")


def test_real_vit_document_gets_table_chunks():
    if not os.path.exists(VIT_MD):
        return
    chunks = process_chunks("vit", VIT_MD)
    tables = {c["section_title"]: c for c in chunks if c["section_title"].startswith("Table ")}
    assert {"Table 1", "Table 2", "Table 3", "Table 4"} <= set(tables)
    assert "0.68k" in tables["Table 2"]["content"] and "9.9k" in tables["Table 2"]["content"]
    assert "307M" in tables["Table 1"]["content"]
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
