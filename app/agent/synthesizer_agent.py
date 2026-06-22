from app.agent.base import BaseAgent, AgentGraphState
from app.models.schemas import AgentResponse
from langgraph.graph import StateGraph, END
from app.utils.logger import logger
from typing import Any, List

class SynthesizerAgent(BaseAgent):
    """
    Agent responsible for synthesizing a concise and focused final answer 
    based on a detailed research analysis and raw context, avoiding verbosity.
    """
    def __init__(self, llm: Any = None):
        super().__init__(
            name="SynthesizerAgent",
            role="Expert editor and synthesizer responsible for converting detailed research analysis into a concise, direct, and well-cited final answer.",
            llm=llm
        )
        
    def _build_graph(self):
        workflow = StateGraph(AgentGraphState)
        workflow.add_node("synthesize", self._synthesize_node)
        workflow.set_entry_point("synthesize")
        workflow.add_edge("synthesize", END)
        return workflow.compile()
        
    def _synthesize_node(self, state: AgentGraphState) -> dict:
        query = state["query"]
        language = state.get("language", "English")
        mode = state.get("mode", "answer")
        findings = state.get("findings", "")
        retrieved_docs = state.get("retrieved_docs", [])
        
        # Build reference logs for the LLM to verify citations (truncated to 1000 chars to avoid context collapse)
        docs_context = ""
        if retrieved_docs:
            docs_context = "\n".join([f"- Document [{i+1}] contents: {doc[:1000]}..." for i, doc in enumerate(retrieved_docs)])

        # Parameterized prompt with strict high-priority directive at the top
        prompt = f"""
        CRITICAL: The final response MUST be written ENTIRELY in {language}. 
        You must answer the User Query directly in {language}.
        Absolutely NO other languages (such as English, Spanish, or Portuguese) are allowed in the output.
        Do not write any introduction or conversational fillers (do not write phrases like "Here is the translation" or "Based on my analysis"). Go directly to the final synthesized answer in {language}.

        You are the SynthesizerAgent, an expert editor. Your objective is to write the ultimate final response to the user's query based strictly on the provided research analysis.
        
        CRITICAL RULES:
        1. DIRECT ANSWERS ONLY: Your response must directly and precisely answer the "User Query". Do not write general summaries, translations of irrelevant documents, or text that does not contribute to answering the query.
        2. SELECTIVE CONTEXT FILTERING: Not all documents or details provided in the "Available Document References" are relevant to the user's query. You must critically filter and use ONLY the context that directly answers the query, ignoring any irrelevant document chunks, figures, or background details.
        3. PRESERVE INDEX CITATIONS: Every claim supported by the research analysis must include its corresponding index citation at the end of the sentence (e.g., [1], [2]). Do not use raw file names.
        4. OUTPUT LANGUAGE: The final response must be written entirely and exclusively in {language}. Translate the relevant English findings to {language}.
        5. NO HALLUCINATION: Do not make up any new facts or claims not supported by the detailed research analysis.
        6. NO RELEVANT INFO HANDLER: If neither the 'Detailed Research Analysis' nor the 'Available Document References' contain enough relevant information to directly answer the User Query, you must state: 'Không tìm thấy thông tin phù hợp trong tài liệu để trả lời câu hỏi này.' (or the equivalent statement in {language}). Do not write a summary of unrelated document contents.
        7. Be concise and direct. Answer the question clearly without unnecessary padding.
        
        User Query:
        "{query}"
        
        Detailed Research Analysis:
        {findings}
        
        Available Document References:
        {docs_context}
        
        Final Answer (written entirely and exclusively in {language}, with index citations like [1]):
        """
        
        try:
            response_text = self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"[{self.name}] LLM synthesis failed: {e}")
            response_text = findings # Fallback to raw findings
            
        return {"final_answer": response_text}
        
    def run(self, query: str, language: str = "English", findings: str = "", retrieved_docs: List[str] = []) -> AgentResponse:
        logger.info(f"=== Starting {self.name} ===")
        initial_state = AgentGraphState(
            query=query,
            language=language,
            findings=findings,
            retrieved_docs=retrieved_docs,
            scratchpad="",
            thought_process=[],
            context_used=[],
            pending_action=None,
            final_answer=None,
            iterations=0
        )
        
        final_state = self.app.invoke(initial_state)
        logger.info(f"=== {self.name} Finished ===")
        
        return AgentResponse(
            thought_process=f"Synthesized research analysis into {language}.",
            content=final_state.get("final_answer") or "No answer.",
            context_used=[],
            retrieved_docs=[]
        )
