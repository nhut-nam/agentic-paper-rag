from typing import Dict, Any, List
from app.pipelines.base import BasePipeline
from app.utils.db import DatabaseHandler
from app.utils.logger import logger
from app.models.chunk import Chunk
from app.llm.embeddings import embedding_service
from keybert import KeyBERT

class RetrievePipeline(BasePipeline):
    """
    Pipeline responsible for retrieving relevant document chunks based on a query.
    Uses Hybrid Search (Vector + Keywords) and Reranking.
    """

    def __init__(self):
        super().__init__()
        self.db = DatabaseHandler()
        # Initialize KeyBERT by reusing the loaded local sentence-transformers model
        logger.info("Initializing KeyBERT for Query Keyword Extraction...")
        self.kw_model = KeyBERT(model=embedding_service.model)

    def run(self, query: str, top_k: int = 5, doc_id: str = None, doc_ids: List[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Executes the retrieval workflow.
        """
        logger.info(f"--- Starting Retrieval for query: {query} ---")

        try:
            # 1. Extract Keywords from Query using local KeyBERT
            logger.info("Extracting keywords from query using local KeyBERT...")
            try:
                keywords_score = self.kw_model.extract_keywords(
                    query,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    top_n=5
                )
                # Ensure they are lowercase and clean
                query_keywords = [kw[0].lower().strip() for kw in keywords_score]
            except Exception as e:
                logger.error(f"KeyBERT query keyword extraction failed: {e}")
                query_keywords = []

            if not query_keywords:
                # Fallback: simple lowercase split if KeyBERT fails
                query_keywords = [w.lower().strip() for w in query.split() if len(w) > 2]

            logger.info(f"Query keywords extracted: {query_keywords}")

            # 2. Generate Embedding for the query using local model
            logger.info("Generating query embedding using local model...")
            query_vector = embedding_service.encode(query)

            # 3. Hybrid Search in DB (Vector Similarity + Keyword Overlap) - Fetch larger candidate pool
            candidate_limit = max(15, top_k)
            logger.info(f"Performing hybrid search (candidates={candidate_limit}, doc_id={doc_id}, doc_ids={doc_ids})...")
            raw_results = self.db.hybrid_search(query_vector, query_keywords, limit=candidate_limit, doc_id=doc_id, doc_ids=doc_ids)

            # 4. Processing candidates
            candidates = []
            for row in raw_results:
                # Convert embedding if it's still a string
                if isinstance(row.get("embedding"), str):
                    vec_str = row["embedding"].strip("[]")
                    row["embedding"] = [float(x) for x in vec_str.split(",")]
                
                candidates.append({
                    "chunk": Chunk(**row),
                    "vector_score": row["vector_score"],
                    "keyword_score": row["keyword_score"],
                    "combined_score": (row["vector_score"] * 0.7 + (row["keyword_score"] * 0.1)) # Normalized keyword score
                })

            # 5. Check Bypass Heuristic: if any chunk has keyword match ratio >= 0.8
            bypass_rerank = False
            total_keywords = len(query_keywords)
            if total_keywords > 0:
                for cand in candidates:
                    ratio = cand["keyword_score"] / total_keywords
                    if ratio >= 0.8:
                        bypass_rerank = True
                        logger.info(f"Bypass Rerank triggered: Candidate {cand['chunk'].chunk_id} matches {cand['keyword_score']}/{total_keywords} keywords (ratio: {ratio:.2f} >= 0.8).")
                        break

            if bypass_rerank or not candidates:
                logger.info("Using hybrid search scores directly (Bypassed Rerank or empty candidates).")
                candidates.sort(key=lambda x: x["combined_score"], reverse=True)
                results = candidates[:top_k]
            else:
                logger.info(f"Reranking {len(candidates)} candidate chunks using Cross-Encoder...")
                try:
                    from app.llm.reranker import reranker_service
                    doc_contents = [c["chunk"].content for c in candidates]
                    rerank_scores = reranker_service.predict(query, doc_contents)
                    
                    # Update scores with semantic rerank scores
                    for cand, score in zip(candidates, rerank_scores):
                        cand["rerank_score"] = score
                        cand["combined_score"] = score
                        
                    # Sort candidates by rerank score
                    candidates.sort(key=lambda x: x["combined_score"], reverse=True)
                    results = candidates[:top_k]
                    logger.info("Reranking completed successfully.")
                except Exception as e:
                    logger.error(f"Reranking failed: {e}. Falling back to hybrid scores.", exc_info=True)
                    candidates.sort(key=lambda x: x["combined_score"], reverse=True)
                    results = candidates[:top_k]

            logger.info(f"Selected top {len(results)} relevant chunks.")
            return results

        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            return []

