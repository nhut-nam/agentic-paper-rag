import os
from typing import Dict, Any
from app.pipelines.base import BasePipeline
from app.services.pdf_handler import process_pdf_to_markdown
from app.utils.logger import logger

class Pdf2MarkdownPipeline(BasePipeline):
    """
    Pipeline responsible for converting PDF documents to structured Markdown.
    """

    def __init__(self):
        super().__init__()

    def _route_handler(self, source_path: str):
        """Routes the file to the appropriate service based on its extension."""
        ext = os.path.splitext(source_path)[1].lower()
        
        if ext == ".pdf":
            return process_pdf_to_markdown
        return None

    def run(self, source_path: str, doc_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        Executes the PDF to Markdown conversion.
        """
        logger.info(f"--- Starting PDF2Markdown: {source_path} ---")
        
        if not os.path.exists(source_path):
            error_msg = f"Source path does not exist: {source_path}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

        handler = self._route_handler(source_path)
        
        if not handler:
            error_msg = f"No handler defined for file extension: {os.path.splitext(source_path)[1]}"
            logger.warning(error_msg)
            return {"status": "skipped", "message": error_msg}

        try:
            # Execute the service function with doc_id
            handler(source_path, doc_id)
            
            logger.info("--- PDF2Markdown Finished Successfully ---")
            return {
                "status": "success",
                "source": source_path,
                "doc_id": doc_id,
                "output_dir": f"processed/{doc_id}"
            }
        except Exception as e:
            logger.error(f"PDF2Markdown failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
