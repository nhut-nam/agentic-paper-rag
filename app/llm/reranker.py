from sentence_transformers import CrossEncoder
from app.config.settings import settings
from app.utils.logger import logger
import torch

class RerankerModel:
    """
    Singleton class for the Reranker Model using sentence-transformers CrossEncoder.
    Ensures the model is loaded into memory (CPU/CUDA) only once.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info(f"Loading reranker model: {settings.RERANKER_MODEL_NAME}...")
            cls._instance = super(RerankerModel, cls).__new__(cls)
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device} for reranker model")
            
            # Load model
            try:
                cls._instance.model = CrossEncoder(
                    settings.RERANKER_MODEL_NAME, 
                    device=device
                )
                logger.info("Reranker model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load reranker model: {e}", exc_info=True)
                raise e
        return cls._instance

    def predict(self, query: str, documents: list[str]) -> list[float]:
        """
        Computes semantic similarity scores for a list of documents against a query.
        Returns a list of float scores representing the relevance of each document.
        """
        if not documents:
            return []
            
        pairs = [[query, doc] for doc in documents]
        try:
            scores = self.model.predict(pairs)
            # Ensure it is a list of floats (if single predict, it could be float; if list, ndarray)
            if hasattr(scores, "tolist"):
                return scores.tolist()
            return [float(s) for s in scores]
        except Exception as e:
            logger.error(f"Reranker prediction failed: {e}", exc_info=True)
            return [0.0] * len(documents)

# Global instance for easy access
reranker_service = RerankerModel()
