import os
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from app.llm.base import BaseLLMProvider
from app.llm.embeddings import embedding_service
from app.utils.logger import logger

class GroqProvider(BaseLLMProvider):
    """
    LLM provider using ChatGroq for LLM and local sentence-transformers for Embeddings.
    """
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("GROQ_API_KEY")
            
        if not api_key:
            logger.error("GROQ_API_KEY is not defined in settings or environment.")
            raise ValueError("GROQ_API_KEY is missing from environment.")
            
        self.llm = ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=0.0
        )
        self.embedding_service = embedding_service
        
    def generate(self, prompt: str, stop: List[str] = None) -> str:
        logger.info(f"--- GROQ GENERATE PROMPT ---\n{prompt}\n----------------------------")
        try:
            if stop:
                response = self.llm.invoke(prompt, stop=stop)
            else:
                response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            return ""

    def generate_summary_and_keywords(self, content: str) -> Dict[str, Any]:
        # Reuse same prompt logic as Ollama if needed, but not used by SynthesizerAgent
        prompt = f"""
        Analyze the following document chunk and provide a brief summary and a list of 5-10 keywords.
        Format your response as a JSON object with keys "summary" and "keywords" (list of strings).
        
        Content:
        {content[:4000]}
        
        JSON Response:
        """
        try:
            response = self.generate(prompt)
            import json
            import re
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                json_str = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', json_str)
                return json.loads(json_str)
            return {"summary": "", "keywords": []}
        except Exception as e:
            logger.error(f"Groq summary/keyword generation failed: {e}")
            return {"summary": "", "keywords": []}

    def get_embeddings(self, text: str) -> List[float]:
        try:
            return self.embedding_service.encode(text)
        except Exception as e:
            logger.error(f"Local embedding generation failed: {e}")
            return []
