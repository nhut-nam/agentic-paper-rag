from typing import List, Optional, Any, TypedDict
from abc import ABC, abstractmethod
from langgraph.graph import StateGraph
from app.models.schemas import Task, StepResult

class AgentGraphState(TypedDict):
    """
    Standard state schema for all LangGraph-based sub-agents.
    """
    query: str
    language: str
    scratchpad: str            # Accumulates Thought, Action, Observation history
    thought_process: List[str] # Explicitly stores thoughts for structured logging/response
    context_used: List[str]    # Records which tools were used and the size/info of the result
    pending_action: Optional[dict] # {"tool": "name", "input": "value"}
    final_answer: Optional[str]
    findings: Optional[str]
    critique: Optional[str]
    iterations: int
    retrieved_docs: List[str]
    is_ambiguous: Optional[bool]

class BaseAgent(ABC):
    def __init__(self, name: str, role: str, llm: Any = None, tools: List[Any] = None, memory: Any = None):
        self.name = name
        self.role = role
        self.llm = llm
        self.tools = tools or []
        self.memory = memory
        
        # Every agent must compile a LangGraph application
        self.app = self._build_graph()
        
    @property
    def system_prompt(self) -> str:
        return f"You are {self.name}, an expert with the following role: {self.role}. Your objective is to successfully complete assigned tasks."

    @abstractmethod
    def _build_graph(self):
        """
        Constructs and compiles the LangGraph StateGraph for this agent.
        Must return a compiled graph (e.g., workflow.compile()).
        """
        pass

    @abstractmethod
    def run(self, query: str, language: str = "English") -> Any:
        """
        Executes the agent's graph for a given query.
        Returns the final result (e.g. AgentResponse).
        """
        pass

    def execute(self, task: Task, context: List[str], language: str = "English") -> StepResult:
        """
        Standard interface for the Orchestrator/Dispatcher to call this agent in the Complex Flow.
        By default, it wraps the task content and context and passes it to the graph's run().
        """
        query_with_context = task.content
        if context:
            query_with_context += f"\n\nContext provided by other agents:\n" + "\n".join(context)
            
        agent_resp = self.run(query_with_context, language=language)
        
        return StepResult(content=agent_resp.content)