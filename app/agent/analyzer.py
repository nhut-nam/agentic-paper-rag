import json
from typing import Optional
from app.models.schemas import QueryComplexity
from app.utils.logger import logger
from app.llm.base import BaseLLMProvider

class ComplexityAnalyzer:
    def __init__(self, llm: BaseLLMProvider):
        """
        Analyzer is a Graph Node utility to evaluate query complexity.
        """
        self.llm = llm
        
    def analyze(self, query: str, session_summary: Optional[str] = None) -> QueryComplexity:
        """
        Uses LLM to evaluate if a query is simple or complex, resolves coreferences/ambiguities using session summary memory,
        and determines if the query remains completely ambiguous/unclear.
        """
        logger.info(f"Analyzer evaluating query: '{query}' with session_summary length: {len(session_summary) if session_summary else 0}")
        
        prompt = f"""
        You are an AI Workflow Router and Query Refiner. Your job is to:
        1. Evaluate the user's query to determine if it relates to document QA/analysis or is a general chit-chat.
        2. Resolve any coreferences or ambiguities in the user's query using the provided session summary memory of the conversation.
        3. Determine if the user's query is completely ambiguous, generic, or incomplete such that it cannot be resolved to a self-contained query (e.g. "Tại sao?", "Làm thế nào?", "Cái gì?", "Mô hình đó là gì?", "Chọn câu nào?" when there is no relevant session summary).
        
        CRITICAL RULES FOR CLASSIFICATION:
        - "mode": "answer" : The query is a direct Q&A about the document context that can be answered quickly. Examples: "What tools does CODEAGENT use?", "Summarize page 3", "Who is the author of this paper?".
        - "mode": "analyze" : The query explicitly requires deep analytical tasks on the document, comparing multiple models, writing comprehensive reports, or multi-step synthesis. Example: "Write a report comparing Swin Transformer and DaViT."
        - "mode": "general" : The query is a general chit-chat, a greeting, or asking general knowledge questions NOT related to the documents in the database. Examples: "Hi, how are you?", "Who are you?", "Tell me a joke", "What is the capital of France?".
        
        COREFERENCE & AMBIGUITY RESOLUTION RULES:
        - Use the Session Summary below to understand the context of the user's current query.
        - If the query contains pronouns (e.g., "it", "they", "this", "câu này", "nó", "mô hình này") or refers to previous items/topics implicitly (e.g. "câu này chọn câu nào" referring to a multiple choice question discussed previously), rewrite/expand it to be a self-contained, clear query.
        - The rewritten query should preserve the original user intent but make all implicit references explicit.
        - If the query is already clear and self-contained, or there is no session summary, "rewritten_query" should be identical to the original query.
        
        AMBIGUITY DETECTION RULES:
        - If the user's query is completely ambiguous, generic, or incomplete such that it cannot be answered or resolved to a self-contained query (e.g. "Tại sao?", "Làm thế nào?", "Cái gì?", "Mô hình đó là gì?", "Chọn câu nào?" when there is no relevant session summary), set "is_ambiguous" to true.
        - If the query is clear or can be successfully resolved using the session summary, set "is_ambiguous" to false.
        
        Session Summary:
        "{session_summary or 'No session history/context available.'}"
        
        User's Current Query: "{query}"
        
        Respond ONLY with a valid JSON object.
        - "mode": string ("answer", "analyze", or "general")
        - "reason": string explaining why based on the rules above.
        - "language": string identifying the language of the user's query (e.g., "English", "Vietnamese", "French").
        - "rewritten_query": string containing the refined/rewritten query with resolved coreferences/ambiguities.
        - "is_ambiguous": boolean (true if the query remains completely ambiguous/unclear, false otherwise)
        
        Examples:
        Query: "câu này chọn câu nào" (with session summary discussing a multiple choice question about Vision Transformer)
        {{"mode": "answer", "reason": "Asking for specific answer choices of the ViT question discussed previously.", "language": "Vietnamese", "rewritten_query": "Trong kiến trúc Vision Transformer (ViT), một hình ảnh được xử lý như thế nào trước khi đưa vào Transformer encoder? Câu này chọn câu nào", "is_ambiguous": false}}
        
        Query: "Cái gì?" (without session summary)
        {{"mode": "general", "reason": "Extremely generic, ambiguous and incomplete query without any session context.", "language": "Vietnamese", "rewritten_query": "Cái gì?", "is_ambiguous": true}}
        
        Query: "Hi, how are you?"
        {{"mode": "general", "reason": "Greeting, out-of-scope chit-chat.", "language": "English", "rewritten_query": "Hi, how are you?", "is_ambiguous": false}}
        """
        
        # Default to general to be safe
        result = QueryComplexity(mode="general", reason="Default fallback", rewritten_query=query, is_ambiguous=False)
        
        try:
            if not self.llm:
                raise ValueError("LLM is not initialized for Analyzer.")
                
            response = self.llm.generate(prompt)
            
            # Extract JSON from LLM response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != -1:
                import re
                json_str = response[start:end]
                # Escape invalid backslashes (like \class) to prevent JSONDecodeError
                json_str = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', json_str)
                data = json.loads(json_str)
                result = QueryComplexity(
                    mode=str(data.get("mode", "analyze")),
                    reason=str(data.get("reason", "Parsed from LLM")),
                    language=str(data.get("language", "English")),
                    rewritten_query=str(data.get("rewritten_query", query)),
                    is_ambiguous=bool(data.get("is_ambiguous", False))
                )
            else:
                logger.error(f"Failed to parse JSON object from LLM response. Raw: {response}")
        except Exception as e:
            logger.error(f"Analyzer LLM generation failed: {e}")
            
        logger.info(f"Analyzer result: mode={result.mode} ({result.reason}), rewritten_query='{result.rewritten_query}', is_ambiguous={result.is_ambiguous}")
        return result
