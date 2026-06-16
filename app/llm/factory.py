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
        
        # Add future providers here:
        # elif provider_type == "openai":
        #     return OpenAIProvider()
            
        raise ValueError(f"Unsupported LLM provider: {provider_type}")
