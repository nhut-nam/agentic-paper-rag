from pydantic import BaseModel
from typing import Optional, Dict, Any

class PipelineResult(BaseModel):
    status: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class IngestResult(PipelineResult):
    source: str
    doc_id: str
    output_dir: str

class ChunkingResult(PipelineResult):
    doc_id: str
    chunks_created: int
    
class ErrorResult(PipelineResult):
    status: str = "error"
    message: str
