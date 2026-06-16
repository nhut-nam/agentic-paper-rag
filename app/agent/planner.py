import uuid
import json
from typing import List, Tuple
from app.models.schemas import Plan, Task, AgentType
from app.utils.logger import logger
from app.llm.base import BaseLLMProvider

class Planner:
    def __init__(self, llm: BaseLLMProvider):
        """
        Planner is a Graph Node utility, not an Agent.
        """
        self.llm = llm
        
    def plan(self, query: str, language: str = "English", mode: str = "answer", 
             global_metadata: str = "", agent_metadata: str = "") -> Tuple[Plan, List[Task]]:
        """
        Uses LLM to break down a query into a Plan and a list of sequential Tasks.
        Assigns an agent to each task based on agent_metadata.
        """
        plan_id = str(uuid.uuid4())
        plan = Plan(id=plan_id, query=query)
        logger.info(f"Planner starting decomposition (Mode: {mode}) for query: '{query}'")
        
        prompt = f"""
        You are the System Planner. Your job is to break down the user's query into a logical sequence of actionable tasks and assign the best agent to each task.
        
        CRITICAL RULES:
        1. The query requires a "{mode}" mode approach. If "analyze", break it down into deep analytical steps. If "answer", keep it to a direct lookup.
        2. Ensure the tasks are generated in the following language: {language}.
        
        AVAILABLE AGENTS:
        {agent_metadata}
        
        GLOBAL CONTEXT (Available Resources):
        {global_metadata}
        
        Query: "{query}"
        
        Respond ONLY with a valid JSON array of objects representing the tasks. Each object must have:
        - "content": A string describing the task clearly.
        - "agent": The name of the agent assigned to this task (e.g., "research").
        
        Example output format:
        [
            {{"content": "Search the local knowledge base for context regarding X.", "agent": "research"}},
            {{"content": "Synthesize the gathered information into a report.", "agent": "synthesis"}}
        ]
        """
        
        tasks_list = []
        try:
            if not self.llm:
                raise ValueError("LLM is not initialized for Planner.")
                
            response = self.llm.generate(prompt)
            
            # Simple JSON extraction
            start = response.find("[")
            end = response.rfind("]") + 1
            if start != -1 and end != -1:
                import re
                json_str = response[start:end]
                # Escape invalid backslashes to prevent JSONDecodeError
                json_str = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', json_str)
                tasks_data = json.loads(json_str)
                
                for idx, t_data in enumerate(tasks_data):
                    agent_str = t_data.get("agent", "research").lower()
                    try:
                        agent_type = AgentType(agent_str)
                    except ValueError:
                        logger.warning(f"Invalid agent type '{agent_str}'. Defaulting to 'research'.")
                        agent_type = AgentType.RESEARCH
                        
                    tasks_list.append(
                        Task(
                            id=str(uuid.uuid4()),
                            plan_id=plan_id,
                            content=t_data.get("content", "Unknown Task"),
                            agent_type=agent_type,
                            task_order=idx + 1
                        )
                    )
            else:
                logger.error(f"Failed to parse JSON array from LLM response. Raw: {response}")
        except Exception as e:
            logger.error(f"Planner LLM generation failed: {e}")
            
        # Fallback if LLM fails
        if not tasks_list:
            logger.warning("Using fallback planning strategy.")
            tasks_list = [
                Task(id=str(uuid.uuid4()), plan_id=plan_id, content=f"Process query: {query}", agent_type=AgentType.RESEARCH, task_order=1)
            ]
        
        logger.info(f"Planner generated {len(tasks_list)} tasks.")
        return plan, tasks_list