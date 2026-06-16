from app.agent.base import BaseAgent, AgentGraphState
from app.models.schemas import AgentResponse
from langgraph.graph import StateGraph, END
from app.utils.logger import logger
from typing import Any

class GeneralAgent(BaseAgent):
    """
    Agent responsible for handling general chit-chat, greetings, and out-of-scope questions.
    Inherits from BaseAgent for structured LangGraph sub-agent conformance.
    """
    def __init__(self, llm: Any = None):
        super().__init__(
            name="GeneralAgent",
            role="Friendly and helpful AI assistant designed to respond to general chit-chat, greetings, and out-of-scope queries naturally and politely.",
            llm=llm
        )
        
    def _build_graph(self):
        """
        Builds a simple single-node graph for quick response generation.
        """
        workflow = StateGraph(AgentGraphState)
        
        workflow.add_node("respond", self._respond_node)
        workflow.set_entry_point("respond")
        workflow.add_edge("respond", END)
        
        return workflow.compile()
        
    def _respond_node(self, state: AgentGraphState) -> dict:
        query = state["query"]
        language = state.get("language", "English")
        is_ambiguous = state.get("is_ambiguous", False)
        
        if is_ambiguous:
            prompt = f"""
            You are called as NamAI, an expert AI assistant specialized in analyzing, reading, and answering questions about scientific papers and research documents.
            
            The user's query is ambiguous, generic, unclear, or lacks key context (even after trying to resolve it using the session history).
            Your job is to politely ask the user to clarify what they mean, specify the target document, or provide more details so that you can help them accurately.
            
            The response MUST be in {language}.
            
            User's Unclear Query: "{query}"
            
            Clarification Request:
            """
        else:
            prompt = f"""
            You are called as NamAI, an expert AI assistant specialized in analyzing, reading, and answering questions about scientific papers and research documents.
            
            Since this query is general chit-chat, a greeting, or general knowledge, respond to the user naturally and politely while maintaining your identity as a specialized Paper Assistant.
            Politely remind or guide the user that you are here to help them read, summarize, analyze, or compare scientific papers in their library, and invite them to ask a question about their documents.
            
            The response MUST be in {language}.
            
            User Query: "{query}"
            
            Response:
            """
        
        try:
            response_text = self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"[{self.name}] LLM generation failed: {e}")
            response_text = "I am sorry, I encountered an error while processing your request."
            
        return {"final_answer": response_text}
        
    def run(self, query: str, language: str = "English", mode: str = "answer", is_ambiguous: bool = False) -> AgentResponse:
        logger.info(f"=== Starting {self.name} Sub-Graph ===")
        initial_state = AgentGraphState(
            query=query,
            language=language,
            mode=mode,
            scratchpad="",
            thought_process=[],
            context_used=[],
            pending_action=None,
            final_answer=None,
            iterations=0,
            retrieved_docs=[],
            is_ambiguous=is_ambiguous
        )
        
        final_state = self.app.invoke(initial_state)
        logger.info(f"=== {self.name} Sub-Graph Finished ===")
        
        return AgentResponse(
            thought_process="Answered directly by GeneralAgent.",
            content=final_state.get("final_answer", "No answer."),
            context_used=[],
            retrieved_docs=[]
        )
