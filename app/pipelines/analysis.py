from typing import Dict, Any
from app.pipelines.base import BasePipeline
from app.services.paper_analyzer import PaperAnalyzer
from app.utils.db import DatabaseHandler
from app.utils.logger import logger

class AnalysisPipeline(BasePipeline):
    """
    Pipeline responsible for taking chunked document data and generating 
    a global, pre-computed analysis (executive summary, methodology, etc.).
    """

    def __init__(self):
        super().__init__()
        self.db = DatabaseHandler()
        self.analyzer = PaperAnalyzer()

    def run(self, doc_id: str, **kwargs) -> Dict[str, Any]:
        """
        Executes the analysis workflow.
        """
        logger.info(f"--- Starting Analysis Pipeline: {doc_id} ---")
        
        # Check if document exists
        doc = self.db.get_document(doc_id)
        if not doc:
            return {"status": "error", "message": "Document not found in DB"}

        try:
            self.db.update_status(doc_id, "analyzing")
            
            # Run the global analysis
            success = self.analyzer.generate_analysis(doc_id)
            
            if not success:
                error_msg = "Failed to generate analysis (LLM or processing error)."
                self.db.update_status(doc_id, f"failed_analysis: {error_msg}")
                return {"status": "error", "message": error_msg}
            
            self.db.update_status(doc_id, "analyzed")
            logger.info(f"--- Analysis Pipeline Finished Successfully: {doc_id} ---")
            
            return {
                "status": "success",
                "doc_id": doc_id,
                "output_dir": f"processed/{doc_id}/analysis.md"
            }

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            self.db.update_status(doc_id, f"failed_analysis: {str(e)}")
            return {"status": "error", "message": str(e)}
