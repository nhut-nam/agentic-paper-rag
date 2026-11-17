import re
import uuid
import os
from typing import List, Dict, Any
from app.utils.logger import logger

class SmartChunker:
    """
    Intelligently splits Markdown content into chunks based on sections (headers).
    Maintains context such as heading paths and section titles.
    """

    def __init__(self, min_chunk_size: int = 100, max_chunk_chars: int = 1500):
        self.min_chunk_size = min_chunk_size
        # all-MiniLM-L6-v2 only reads ~256 word pieces (~1,000-1,200 chars of English).
        # Longer sections were embedded from their first paragraph only, so we split them.
        self.max_chunk_chars = max_chunk_chars

    # ------------------------------------------------------------------
    # Long sections -> several chunks (split on paragraph boundaries)
    # ------------------------------------------------------------------
    def split_long_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        content = chunk["content"]
        if len(content) <= self.max_chunk_chars:
            return [chunk]
        paragraphs = [p for p in re.split(r"\n\s*\n", content) if p.strip()]
        parts, current = [], ""
        for para in paragraphs:
            if current and len(current) + len(para) + 2 > self.max_chunk_chars:
                parts.append(current)
                current = para
            else:
                current = f"{current}\n\n{para}" if current else para
        if current:
            parts.append(current)
        if len(parts) == 1:
            return [chunk]
        return [
            {**chunk, "content": part, "section_title": f"{chunk['section_title']} (part {i}/{len(parts)})"}
            for i, part in enumerate(parts, start=1)
        ]

    # ------------------------------------------------------------------
    # Tables that marker rendered as images -> their own searchable chunks
    # ------------------------------------------------------------------
    _IMAGE_LINK = re.compile(r"!\[[^\]]*\]\(\.\./images/([^/)]+)/image\.png\)")
    _CAPTION = re.compile(r"(Table\s+\d+\s*:.*)")
    _SPAN = re.compile(r"<span[^>]*></span>")

    def build_table_chunks(self, markdown_content: str, images_dir: str) -> List[Dict[str, Any]]:
        """
        For every image link whose saved content.md is a markdown table, create a chunk
        "<caption>\n\n<table>" so the numbers are embedded and keyword-searchable
        (before, table text was only attached at answer time and often cut off).
        """
        lines = markdown_content.splitlines()
        header_pattern = re.compile(r'^(#{1,6})\s+(.*)$')
        headings = {1: "", 2: "", 3: "", 4: "", 5: "", 6: ""}
        chunks = []
        for idx, line in enumerate(lines):
            h = header_pattern.match(line)
            if h:
                level = len(h.group(1))
                headings[level] = self._SPAN.sub("", h.group(2)).strip()
                for l in range(level + 1, 7):
                    headings[l] = ""
                continue
            m = self._IMAGE_LINK.search(line)
            if not m:
                continue
            image_id = m.group(1)
            table_path = os.path.join(images_dir, image_id, "content.md")
            if not os.path.exists(table_path):
                continue
            with open(table_path, "r", encoding="utf-8") as f:
                table_md = self._SPAN.sub("", f.read()).strip()
            if not table_md.startswith("|"):
                continue  # equations / code, not tables
            caption = self._find_caption(lines, idx)
            label = caption.split(":")[0].strip() if caption else f"Table ({image_id})"
            section_path = " > ".join(v for v in headings.values() if v) or "Root"
            chunks.append({
                "content": f"{caption or label}\n\n{table_md}\n\n(Source image: ../images/{image_id}/image.png)",
                "heading_path": f"{section_path} > {label}",
                "section_title": label,
            })
        return chunks

    def _find_caption(self, lines: List[str], idx: int, window: int = 4) -> str:
        """Captions sit just after (sometimes just before) the image line."""
        order = list(range(idx + 1, min(len(lines), idx + 1 + window))) + list(range(idx - 1, max(-1, idx - 1 - window), -1))
        for j in order:
            if self._IMAGE_LINK.search(lines[j]):
                break  # reached another image: stop looking in this direction
            m = self._CAPTION.search(self._SPAN.sub("", lines[j]))
            if m:
                return m.group(1).strip()
        return ""

    def chunk_by_headers(self, markdown_content: str) -> List[Dict[str, Any]]:
        """
        Splits markdown by headers (#, ##, etc.) and returns a list of chunks.
        """
        # Regex to find headers
        header_pattern = re.compile(r'^(#{1,6})\s+(.*)$', re.MULTILINE)
        
        chunks = []
        last_pos = 0
        current_headings = {1: "", 2: "", 3: "", 4: "", 5: "", 6: ""}
        
        # Find all headers
        matches = list(header_pattern.finditer(markdown_content))
        
        if not matches:
            # No headers found, treat entire content as one chunk
            return [{
                "content": markdown_content.strip(),
                "heading_path": "Root",
                "section_title": "Root"
            }]

        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            
            # Extract content between previous header and current header
            if i > 0:
                prev_match = matches[i-1]
                content = markdown_content[prev_match.end():match.start()].strip()
                
                if len(content) >= self.min_chunk_size or i == 1:
                    chunks.append({
                        "content": content,
                        "heading_path": self._build_path(current_headings, prev_match_level=len(prev_match.group(1))),
                        "section_title": prev_match.group(2).strip()
                    })

            # Update current heading path after processing the previous content chunk
            current_headings[level] = title
            # Reset lower level headings
            for l in range(level + 1, 7):
                current_headings[l] = ""

        # Add the last section
        last_match = matches[-1]
        last_content = markdown_content[last_match.end():].strip()
        chunks.append({
            "content": last_content,
            "heading_path": self._build_path(current_headings, prev_match_level=len(last_match.group(1))),
            "section_title": last_match.group(2).strip()
        })

        return chunks

    def _build_path(self, headings: Dict[int, str], prev_match_level: int) -> str:
        """Constructs a string path from the current heading state."""
        path_parts = []
        for l in range(1, prev_match_level + 1):
            if headings[l]:
                path_parts.append(headings[l])
        return " > ".join(path_parts) if path_parts else "Root"

def process_chunks(doc_id: str, markdown_path: str) -> List[Dict[str, Any]]:
    """
    Helper function to read a file and produce chunks ready for the database.
    """
    if not os.path.exists(markdown_path):
        logger.error(f"Markdown file not found: {markdown_path}")
        return []

    with open(markdown_path, 'r', encoding='utf-8') as f:
        content = f.read()

    chunker = SmartChunker()
    raw_chunks = []
    for rc in chunker.chunk_by_headers(content):
        raw_chunks.extend(chunker.split_long_chunk(rc))

    # processed/{doc_id}/markdown/result.md -> processed/{doc_id}/images/
    images_dir = os.path.join(os.path.dirname(os.path.dirname(markdown_path)), "images")
    table_chunks = chunker.build_table_chunks(content, images_dir) if os.path.isdir(images_dir) else []
    if table_chunks:
        logger.info(f"Created {len(table_chunks)} table chunks from extracted table images.")
    raw_chunks.extend(table_chunks)
    
    final_chunks = []
    for i, rc in enumerate(raw_chunks):
        final_chunks.append({
            "chunk_id": f"{doc_id}_ch_{i}",
            "doc_id": doc_id,
            "content": rc["content"],
            "heading_path": rc["heading_path"],
            "section_title": rc["section_title"],
            "chunk_order": i
        })
        
    return final_chunks
