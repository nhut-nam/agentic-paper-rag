from app.llm.base import BaseLLMProvider
from app.llm.ollama_provider import OllamaProvider
from app.config.settings import settings

class LLMFactory:
    """
    Factory to create the appropriate LLM provider based on settings.
    """
    
    @staticmethod
    def get_provider(provider_type: str = "ollama") -> BaseLLMProvider:
        """
        Returns an instance of the requested LLM provider.
        """
        provider_type = provider_type.lower()
        
        if provider_type == "ollama":
            return OllamaProvider()
        elif provider_type == "groq":
            from app.llm.groq_provider import GroqProvider
            return GroqProvider()
            
        raise ValueError(f"Unsupported LLM provider: {provider_type}")
