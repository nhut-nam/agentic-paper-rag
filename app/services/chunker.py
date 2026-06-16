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

    def __init__(self, min_chunk_size: int = 100):
        self.min_chunk_size = min_chunk_size

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
    raw_chunks = chunker.chunk_by_headers(content)
    
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
