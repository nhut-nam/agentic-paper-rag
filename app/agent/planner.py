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
        
    def plan(self, query: str, language: str = "English",
             global_metadata: str = "", agent_metadata: str = "") -> Tuple[Plan, List[Task]]:
        """
        Uses LLM to break down a query into a Plan and a list of sequential Tasks.
        Assigns an agent to each task based on agent_metadata.
        """
        plan_id = str(uuid.uuid4())
        plan = Plan(id=plan_id, query=query)
        logger.info(f"Planner starting decomposition for query: '{query}'")
        
        prompt = f"""
        You are the System Planner. Your job is to break down the user's query into a logical sequence of actionable tasks and assign the best agent to each task.
        
        CRITICAL RULES:
        1. Generate the MINIMUM number of tasks needed to answer the query:
           - For simple lookups: EXACTLY 1 task.
           - For complex multi-part queries: maximum 2-3 tasks.
        2. FORBIDDEN TASKS — Do NOT generate tasks that:
           - Translate, rephrase, or summarize the query itself.
           - Are meta-tasks (e.g., "Understand the query", "Translate to English").
        3. Each task MUST start with an action verb: Search, Find, Retrieve, Analyze, Compare.
        4. Task content MUST be written in English.
        
        AVAILABLE AGENTS:
        {agent_metadata}
        
        GLOBAL CONTEXT (Available Resources):
        {global_metadata}
        
        Query: "{query}"
        
        Respond ONLY with a valid JSON array of objects. Each object must have:
        - "content": A specific research instruction in English.
        - "agent": The name of the agent (always use "research").
        
        Example output for an "answer" mode query:
        [
            {{"content": "Search the local knowledge base for the optimizer and batch size used during ViT pre-training on JFT-300M.", "agent": "research"}}
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