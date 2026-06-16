from typing import Dict, Any
from app.pipelines.base import BasePipeline
from app.pipelines.pdf2markdown import Pdf2MarkdownPipeline
from app.pipelines.chunking import ChunkingPipeline
from app.utils.db import DatabaseHandler
from app.utils.logger import logger

class IngestPipeline(BasePipeline):
    """
    Master Ingestion Pipeline.
    Orchestrates: PDF to Markdown -> Smart Chunking -> LLM Enrichment -> Vector Store (DB).
    """

    def __init__(self):
        super().__init__()
        self.db = DatabaseHandler()

    def run(self, source_path: str, doc_id: str, **kwargs) -> Dict[str, Any]:
        """
        Executes the full end-to-end ingestion workflow.
        """
        logger.info(f"===== STARTING MASTER INGESTION: {doc_id} =====")
        
        try:
            # 1. PDF to Markdown
            self.db.update_status(doc_id, "ingesting")
            pdf2md = Pdf2MarkdownPipeline()
            md_result = pdf2md.run(source_path, doc_id=doc_id)
            
            if md_result["status"] != "success":
                self.db.update_status(doc_id, f"failed_md: {md_result.get('message')}")
                return md_result

            # 2. Smart Chunking & LLM Enrichment
            self.db.update_status(doc_id, "chunking")
            chunking = ChunkingPipeline()
            chunk_result = chunking.run(doc_id)
            
            if chunk_result["status"] != "success":
                self.db.update_status(doc_id, f"failed_chunking: {chunk_result.get('message')}")
                return chunk_result

            # Final Success
            self.db.update_status(doc_id, "chunked")
            logger.info(f"===== MASTER INGESTION COMPLETED: {doc_id} =====")
            
            return {
                "status": "success",
                "doc_id": doc_id,
                "message": "Full ingestion and knowledge base creation completed."
            }

        except Exception as e:
            logger.error(f"Master Ingestion failed for {doc_id}: {e}", exc_info=True)
            self.db.update_status(doc_id, f"failed_master: {str(e)}")
            return {"status": "error", "message": str(e)}
