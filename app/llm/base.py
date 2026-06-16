from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM and Embedding providers.
    """
    
    @abstractmethod
    def generate(self, prompt: str, stop: List[str] = None) -> str:
        """Generates raw text from the given prompt."""
        pass
        

    @abstractmethod
    def generate_summary_and_keywords(self, content: str) -> Dict[str, Any]:
        """Generates summary and keywords for the given content."""
        pass

    @abstractmethod
    def get_embeddings(self, text: str) -> List[float]:
        """Generates embeddings for the given text."""
        pass
