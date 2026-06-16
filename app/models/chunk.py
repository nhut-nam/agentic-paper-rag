from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ChunkBase(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    summary: Optional[str] = None
    keywords: List[str] = []
    heading_path: Optional[str] = None
    section_title: Optional[str] = None
    page_ref: Optional[str] = None
    chunk_order: Optional[int] = None
    embedding: Optional[List[float]] = None

class ChunkCreate(ChunkBase):
    pass

class Chunk(ChunkBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True
