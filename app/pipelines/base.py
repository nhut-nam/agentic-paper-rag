from abc import ABC, abstractmethod
from typing import Any, Dict

class BasePipeline(ABC):
    """
    Abstract base class for all pipelines in the system.
    Provides a consistent interface for orchestration and Agent integration.
    """
    
    @abstractmethod
    def run(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Execute the pipeline logic.
        
        Args:
            input_data: The primary input for the pipeline (e.g., file path, query string).
            **kwargs: Additional configuration or context.
            
        Returns:
            A dictionary containing the results, metadata, and status.
        """
        pass

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
