from sentence_transformers import SentenceTransformer
from app.config.settings import settings
from app.utils.logger import logger
import torch

class EmbeddingModel:
    """
    Singleton class for the Embedding Model using sentence-transformers.
    Ensures the model is loaded into memory only once.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}...")
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device} for embeddings")
            
            # Load model
            cls._instance.model = SentenceTransformer(
                settings.EMBEDDING_MODEL_NAME, 
                device=device
            )
            logger.info("Embedding model loaded successfully.")
        return cls._instance

    def encode(self, text: str):
        """Generates embedding for the given text."""
        return self.model.encode(text).tolist()

# Global instance for easy access
embedding_service = EmbeddingModel()
