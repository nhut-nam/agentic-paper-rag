from typing import Dict, Any
import os
from app.pipelines.base import BasePipeline
from app.services.chunker import process_chunks
from app.utils.db import DatabaseHandler
from app.utils.logger import logger
from app.utils.storage import StorageManager
from app.llm.factory import LLMFactory
from keybert import KeyBERT
from app.llm.embeddings import embedding_service

class ChunkingPipeline(BasePipeline):
    """
    Pipeline responsible for taking an ingested document and splitting it into smart chunks,
    then enhancing them with keywords (via KeyBERT) and embeddings from raw text.
    """

    def __init__(self):
        super().__init__()
        self.db = DatabaseHandler()
        self.llm_service = LLMFactory.get_provider("ollama")
        # Load KeyBERT by reusing the already loaded local sentence-transformers model
        logger.info("Initializing KeyBERT with existing local embedding model...")
        self.kw_model = KeyBERT(model=embedding_service.model)

    def extract_keywords(self, text: str) -> list:
        """Extracts keyphrases/keywords from the text using KeyBERT locally."""
        try:
            keywords_score = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=8
            )
            return [kw[0] for kw in keywords_score]
        except Exception as e:
            logger.error(f"KeyBERT keyword extraction failed: {e}")
            return []

    def run(self, doc_id: str, **kwargs) -> Dict[str, Any]:
        """
        Executes the chunking workflow.
        """
        logger.info(f"--- Starting Chunking Pipeline: {doc_id} ---")
        
        # 1. Get document info
        doc = self.db.get_document(doc_id)
        if not doc:
            return {"status": "error", "message": "Document not found in DB"}

        storage = StorageManager()
        markdown_path = storage.get_full_path(f"processed/{doc_id}/markdown/result.md")

        if not os.path.exists(markdown_path):
            return {"status": "error", "message": f"Markdown file not found at {markdown_path}"}

        try:
            # 2. Perform smart chunking
            self.db.update_status(doc_id, "chunking")
            chunks = process_chunks(doc_id, markdown_path)
            
            # 3. Enhance chunks and Store in DB
            logger.info(f"Enhancing {len(chunks)} chunks with KeyBERT keywords and raw text embeddings...")
            for chunk in chunks:
                # Extract keywords using KeyBERT locally
                chunk["keywords"] = self.extract_keywords(chunk["content"])
                
                # Use a preview of raw content as the summary
                content_len = len(chunk["content"])
                chunk["summary"] = chunk["content"][:200] + "..." if content_len > 200 else chunk["content"]
                
                # Generate embeddings based directly on Raw Content (Semantic Precision)
                chunk["embedding"] = self.llm_service.get_embeddings(chunk["content"])
                
                # Save to DB
                self.db.insert_chunk(chunk)
            
            self.db.update_status(doc_id, "chunked")
            logger.info(f"Successfully processed and stored {len(chunks)} chunks for document {doc_id}")
            
            return {
                "status": "success",
                "doc_id": doc_id,
                "chunks_created": len(chunks)
            }
        except Exception as e:
            logger.error(f"Chunking failed: {e}", exc_info=True)
            self.db.update_status(doc_id, f"failed_chunking: {str(e)}")
            return {"status": "error", "message": str(e)}
