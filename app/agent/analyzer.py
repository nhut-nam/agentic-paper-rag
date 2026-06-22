import json
from typing import Optional
from app.models.schemas import QueryComplexity
from app.utils.logger import logger
from app.llm.base import BaseLLMProvider

class ComplexityAnalyzer:
    def __init__(self, llm: BaseLLMProvider):
        """
        Analyzer is a Graph Node utility.
        Determines if a query is ambiguous/chitchat, detects language,
        and resolves coreferences using session context.
        No longer classifies 'mode' — routing is done purely by is_ambiguous.
        """
        self.llm = llm
        
    def analyze(self, query: str, session_summary: Optional[str] = None) -> QueryComplexity:
        """
        Evaluates a query and returns:
        - language: detected language
        - rewritten_query: coreference-resolved, self-contained query
        - is_ambiguous: True if query cannot be answered (pure chitchat or too vague)
        """
        logger.info(f"Analyzer evaluating query: '{query}' | session_summary length: {len(session_summary) if session_summary else 0}")
        
        prompt = f"""You are a Query Analyzer. Given a user's query and the conversation history, do two things:
1. Resolve any pronouns or implicit references using the Session Summary so the query becomes self-contained.
2. Decide if the query is answerable (needs research/lookup) or is pure chit-chat/completely ambiguous.

RULES:
- Set "is_ambiguous" to true ONLY if the query is clearly chit-chat (e.g. "Hi", "How are you?", "Tell me a joke") or is completely unclear with no context to resolve it (e.g. "Why?", "What?" with no session history).
- Set "is_ambiguous" to false for any question about a document, paper, concept, data, or topic — even if complex.
- In "rewritten_query": rewrite to make implicit references explicit using the session summary. If the query is already clear, copy it unchanged.
- In "language": identify the language of the user's query (e.g. "Vietnamese", "English").

Session Summary:
"{session_summary or 'No prior conversation.'}"

User Query: "{query}"

Respond ONLY with valid JSON (no extra text):
{{"language": "<language>", "rewritten_query": "<resolved query>", "is_ambiguous": <true|false>, "reason": "<one sentence explanation>"}}

Examples:
Query: "câu này chọn câu nào" | Session: discussing ViT multiple-choice question
{{"language": "Vietnamese", "rewritten_query": "Trong bài trắc nghiệm về ViT đang thảo luận, đáp án đúng là câu nào?", "is_ambiguous": false, "reason": "Coreference resolved using session context."}}

Query: "Hi, how are you?"
{{"language": "English", "rewritten_query": "Hi, how are you?", "is_ambiguous": true, "reason": "Pure greeting, not a research question."}}

Query: "What optimizer did ViT use for pre-training?"
{{"language": "English", "rewritten_query": "What optimizer did ViT use for pre-training?", "is_ambiguous": false, "reason": "Clear research question about ViT paper."}}"""

        # Default: treat as non-ambiguous research query
        result = QueryComplexity(reason="Default fallback", rewritten_query=query, is_ambiguous=False)
        
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
                
                data = {}
                try:
                    clean_str = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', json_str)
                    data = json.loads(clean_str)
                except Exception as json_err:
                    logger.warning(f"Standard JSON parse failed ({json_err}). Using regex extraction.")
                    # Robust regex fallback
                    lang_m = re.search(r'"language"\s*:\s*"([^"]*?)"', json_str)
                    if lang_m: data["language"] = lang_m.group(1)
                    
                    ambig_m = re.search(r'"is_ambiguous"\s*:\s*(true|false)', json_str, re.IGNORECASE)
                    if ambig_m: data["is_ambiguous"] = ambig_m.group(1).lower() == "true"
                    
                    reason_m = re.search(r'"reason"\s*:\s*"([^"]*?)"', json_str)
                    if reason_m: data["reason"] = reason_m.group(1)
                    
                    # rewritten_query may span multiple words with potential quotes
                    q_m = re.search(r'"rewritten_query"\s*:\s*"(.*?)"(?:\s*,|\s*})', json_str, re.DOTALL)
                    if q_m: data["rewritten_query"] = q_m.group(1)

                result = QueryComplexity(
                    reason=str(data.get("reason", "Parsed from LLM")),
                    language=str(data.get("language", "English")),
                    rewritten_query=str(data.get("rewritten_query", query)),
                    is_ambiguous=bool(data.get("is_ambiguous", False))
                )
            else:
                logger.error(f"Failed to parse JSON from LLM response. Raw: {response}")
                
        except Exception as e:
            logger.error(f"Analyzer LLM generation failed: {e}")
            
        logger.info(f"Analyzer result: is_ambiguous={result.is_ambiguous} | lang={result.language} | rewritten='{result.rewritten_query}'")
        return result
