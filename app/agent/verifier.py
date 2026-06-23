import json
import re
from typing import Dict, Any, Optional
from app.utils.logger import logger
from app.llm.base import BaseLLMProvider

class ContextVerifier:
    def __init__(self, llm: BaseLLMProvider):
        """
        Verifier is a Graph Node utility.
        Determines if the retrieved context is sufficient to answer the user query.
        """
        self.llm = llm

    def verify(self, query: str, context: str) -> Dict[str, Any]:
        """
        Evaluates a query against the retrieved context to verify sufficiency.
        Returns a dict:
        - "sufficient": bool
        - "feedback": str
        """
        logger.info(f"ContextVerifier checking context sufficiency for query: '{query}'")
        
        # Build evaluation prompt
        prompt = f"""You are a strict Retrieval-Augmented Generation (RAG) Context Verifier. 
Your objective is to evaluate whether the retrieved context contains enough details, facts, numbers, or visual descriptions to directly and accurately answer the user's query.

User Query:
"{query}"

Retrieved Context:
\"\"\"
{context}
\"\"\"

CRITICAL RULES FOR EVALUATION:
1. BE EXTRA CRITICAL: If the query asks for specific numbers, hyperparameters, optimization settings, metrics, tables, figures, or results, and they are NOT explicitly mentioned in the context (or are too vague/generic), you MUST mark the context as INSUFFICIENT ("sufficient": false).
2. MULTIPLE PARTS: If the query has multiple parts (e.g. "What optimizer was used, and what was the batch size?"), and only one part is answered in the context, you MUST mark it as INSUFFICIENT.
3. IMAGES & FIGURES: If the query asks about details illustrated in a specific figure or table, and the text context only mentions the figure/table but doesn't describe the actual data, mark it as INSUFFICIENT and request analyzing that specific figure/table image.
4. ACTIONABLE FEEDBACK & KEYWORDS: If the context is insufficient, specify exactly what is missing in the "feedback" field, and provide a list of 4-6 highly specific technical search terms (unigrams or short phrases, e.g. ["Adam", "4096", "Table 3", "optimizer"]) in the "suggested_keywords" field. 
5. NO LOCAL CITATION SYMBOLS: Do NOT include dynamic document indices or citation markers such as "Document [x]", "Document [number]", or "Source: ..." in either "feedback" or "suggested_keywords".
6. ONLY JSON: Respond strictly with a JSON object. Do not include any explanation or extra text outside the JSON.

Expected JSON format:
{{"sufficient": <true|false>, "feedback": "<what is missing>", "suggested_keywords": ["keyword1", "keyword2", ...]}}
"""

        # Default fallback
        result = {"sufficient": True, "feedback": "Default fallback (sufficient)", "suggested_keywords": []}
        
        try:
            if not self.llm:
                raise ValueError("LLM is not initialized for ContextVerifier.")
                
            response = self.llm.generate(prompt)
            logger.info(f"ContextVerifier Raw LLM Response: {response}")
            
            # Extract JSON from LLM response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                # Escape invalid backslashes to prevent JSONDecodeError
                json_str = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', json_str)
                
                try:
                    data = json.loads(json_str)
                    # Standardize keys
                    sufficient = data.get("sufficient")
                    if isinstance(sufficient, str):
                        sufficient = sufficient.lower() == "true"
                    elif not isinstance(sufficient, bool):
                        sufficient = bool(sufficient)
                        
                    feedback = str(data.get("feedback", "No feedback provided."))
                    suggested_keywords = data.get("suggested_keywords", [])
                    if not isinstance(suggested_keywords, list):
                        suggested_keywords = []
                    
                    result = {
                        "sufficient": sufficient,
                        "feedback": feedback,
                        "suggested_keywords": [str(k) for k in suggested_keywords]
                    }
                except Exception as json_err:
                    logger.warning(f"Standard JSON parse failed in verifier ({json_err}). Using regex extraction.")
                    # Regex extraction fallback
                    suff_match = re.search(r'"sufficient"\s*:\s*(true|false)', json_str, re.IGNORECASE)
                    feed_match = re.search(r'"feedback"\s*:\s*"(.*?)"(?:\s*,|\s*})', json_str, re.DOTALL)
                    
                    sufficient = True
                    if suff_match:
                        sufficient = suff_match.group(1).lower() == "true"
                        
                    feedback = feed_match.group(1) if feed_match else "Context is insufficient."
                    
                    # Regex for suggested_keywords array
                    kw_match = re.search(r'"suggested_keywords"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
                    suggested_keywords = []
                    if kw_match:
                        kws_str = kw_match.group(1)
                        suggested_keywords = [k.strip(' \'"') for k in kws_str.split(',') if k.strip()]
                        
                    result = {
                        "sufficient": sufficient,
                        "feedback": feedback,
                        "suggested_keywords": suggested_keywords
                    }
            else:
                logger.error(f"Failed to find JSON block in Verifier response. Raw: {response}")
                result = {"sufficient": False, "feedback": f"Could not verify. Retrieve more general information about: {query}", "suggested_keywords": []}
                
        except Exception as e:
            logger.error(f"ContextVerifier generation failed: {e}", exc_info=True)
            # Default fallback on LLM failure is to assume it's sufficient to prevent infinite retries
            result = {"sufficient": True, "feedback": "Verifier failed, fallback to sufficient.", "suggested_keywords": []}
            
        logger.info(f"ContextVerifier result: {result}")
        return result
