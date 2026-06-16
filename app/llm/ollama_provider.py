import json
from typing import List, Dict, Any
from langchain_ollama import OllamaLLM
from app.llm.base import BaseLLMProvider
from app.llm.embeddings import embedding_service
from app.config.settings import settings
from app.utils.logger import logger

class OllamaProvider(BaseLLMProvider):
    """
    Implementation of LLM provider using Ollama for LLM and local transformers for Embeddings.
    """
    
    def __init__(self):
        self.llm = OllamaLLM(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_LLM_MODEL
        )
        self.embedding_service = embedding_service

    def generate(self, prompt: str, stop: List[str] = None) -> str:
        logger.info(f"--- LLM GENERATE PROMPT ---\n{prompt}\n---------------------------")
        try:
            if stop:
                return self.llm.invoke(prompt, stop=stop)
            return self.llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return ""

    def generate_summary_and_keywords(self, content: str) -> Dict[str, Any]:
        prompt = f"""
        Analyze the following document chunk and provide a brief summary and a list of 5-10 keywords.
        Format your response as a JSON object with keys "summary" and "keywords" (list of strings).
        
        Content:
        {content[:4000]}
        
        JSON Response:
        """
        logger.info(f"--- LLM SUMMARY PROMPT ---\n{prompt}\n--------------------------")
        try:
            response = self.llm.invoke(prompt)
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != -1:
                import re
                json_str = response[start:end]
                # Escape invalid backslashes to prevent JSONDecodeError
                json_str = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', json_str)
                return json.loads(json_str)
            return {"summary": "", "keywords": []}
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return {"summary": "", "keywords": []}

    def get_embeddings(self, text: str) -> List[float]:
        try:
            return self.embedding_service.encode(text)
        except Exception as e:
            logger.error(f"Local embedding generation failed: {e}")
            return []
