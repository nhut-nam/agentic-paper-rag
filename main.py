import os
import sys
from app.pipelines.ingest import IngestPipeline
from app.utils.logger import logger

def main():
    """
    Main entry point for the Paper Intelligent AI application.
    """
    target_file = "paper.pdf"
    
    try:
        # Initialize the pipeline
        pipeline = IngestPipeline()
        
        # Execute the pipeline with a default doc_id for direct local testing
        doc_id = "default_local_paper"
        
        # Register the document in DB if not exists so that status updates and chunking work
        from app.utils.db import DatabaseHandler
        db = DatabaseHandler()
        db.insert_document(doc_id, target_file)
        
        result = pipeline.run(target_file, doc_id=doc_id)
        
        if result["status"] == "success":
            logger.info(f"Ingestion successful! Results: {result}")
        else:
            logger.error(f"Ingestion failed: {result.get('message')}")
            
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
