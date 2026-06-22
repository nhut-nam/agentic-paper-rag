from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    CANCELLED = "cancelled"
    INSUFFICIENT = "insufficient"


class AgentType(str, Enum):
    RESEARCH = "research"
    WEB = "web"
    VISION = "vision"
    SYNTHESIS = "synthesis"
    PLANNER = "planner"

class QueryComplexity(BaseModel):
    reason: str = ""
    language: str = "English"
    rewritten_query: Optional[str] = None
    is_ambiguous: bool = False

class AgentResponse(BaseModel):
    thought_process: str
    content: str
    context_used: List[str] = Field(default_factory=list)
    retrieved_docs: List[str] = Field(default_factory=list)


class Plan(BaseModel):
    id: str
    query: str


class Task(BaseModel):
    id: str
    plan_id: str
    content: str
    agent_type: AgentType
    task_order: int


class ContextRequest(BaseModel):
    from_agent: str
    need: str
    reason: str
    retry_with: AgentType


class StepResult(BaseModel):
    content: str
    context_request: Optional[ContextRequest] = None


class Step(BaseModel):
    id: str
    task_id: str
    agent_type: AgentType
    status: StepStatus = StepStatus.PENDING
    result: Optional[StepResult] = None
    context: List[str] = Field(default_factory=list)