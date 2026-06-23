import re
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from app.models.schemas import AgentResponse
from app.agent.base import BaseAgent, AgentGraphState
from app.utils.logger import logger
from app.tools import get_retrieve_tool, web_search_tool, get_vision_tool
from app.config.settings import settings

class ResearchAgent(BaseAgent):
    """
    Agent responsible for conducting research using a custom LangGraph ReAct Sub-Graph.
    """
    def __init__(self, llm=None, doc_id=None, doc_ids=None):
        llm_instance = llm or OllamaLLM(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_LLM_MODEL)
        
        # Merge doc_id and doc_ids
        target_ids = []
        if doc_ids:
            target_ids.extend(doc_ids)
        if doc_id and doc_id not in target_ids:
            target_ids.append(doc_id)
            
        # Inject target_ids dynamically to retrieve and vision tools
        scoped_retrieve_tool = get_retrieve_tool(doc_ids=target_ids)
        scoped_vision_tool = get_vision_tool(doc_ids=target_ids)
        tools_list = [scoped_retrieve_tool, web_search_tool, scoped_vision_tool]
        
        super().__init__(
            name="ResearchAgent",
            role="Expert researcher capable of finding information from the database, searching the web, and analyzing images to answer queries. You MUST strictly rely on tool outputs and NEVER make up information.",
            llm=llm_instance,
            tools=tools_list
        )
        
        # Build tool dictionary mapping name to tool function
        self.tool_map = {tool.name: tool for tool in self.tools}

    def _build_graph(self):
        """
        Constructs the Custom ReAct LangGraph.
        """
        workflow = StateGraph(AgentGraphState)
        
        workflow.add_node("reasoning", self._reasoning_node)
        workflow.add_node("tool", self._tool_node)
        
        workflow.set_entry_point("reasoning")
        
        workflow.add_conditional_edges(
            "reasoning",
            self._route_after_reasoning,
            {
                "tool": "tool",
                "end": END
            }
        )
        
        workflow.add_edge("tool", "reasoning")
        
        return workflow.compile()

    def _reasoning_node(self, state: AgentGraphState) -> dict:
        query = state["query"]
        scratchpad = state.get("scratchpad", "")
        iterations = state.get("iterations", 0)
        
        tool_desc = "\n".join([f"- {t.name}: {t.description}" for t in self.tools])
        tool_names = ", ".join([t.name for t in self.tools])
        
        # Define loop warning for final iteration (iteration 3)
        loop_warning = ""
        if iterations == 3:
            loop_warning = "\nWARNING: This is your final iteration! You must synthesize the observations from the previous steps and provide the Final Answer now. You are NOT allowed to take another action."
            
        prompt = f"""{self.system_prompt}


You have access to the following tools:
{tool_desc}

CRITICAL RULES:
1. STRICTLY FORBIDDEN TO HALLUCINATE: You must ONLY use the information provided by the tools. If the tools do not contain the answer, explicitly say "I do not have enough information to answer this."
2. ALWAYS CITE SOURCES USING NUMERICAL INDEXES: The retrieve_tool returns document chunks with index headers (e.g., --- Document [1] ... ---). In your Final Answer, every single factual claim, core data, or important definition MUST include an inline numerical citation index at the end of the sentence indicating exactly which document it came from (e.g., [1], [2]). Avoid using raw file names or paths in citations. If multiple documents support a claim, list them all (e.g., [1, 2]). Do not write any important statement without citing its index.
3. MANDATORY IMAGE ANALYSIS: In the document chunks, images are represented as relative links like `../images/{{image_id}}/image.png`. If a chunk contains an image link and the user's question asks about visual details, charts, figures, tables, or if the text indicates that details are illustrated in a figure, you MUST explicitly call the `vision_tool` with that exact image link (e.g., `../images/{{image_id}}/image.png`) to extract and understand the visual content. Do not guess or ignore images.
4. EXACT TOOL NAMES: The 'Action:' field MUST contain exactly one of the tool names: [{tool_names}]. Do not translate the tool name or add extra spaces. Example: Action: retrieve_tool
5. STOP AFTER ACTION INPUT: You are the THINKER, not the executor. Once you generate 'Action Input:', you MUST STOP. DO NOT generate 'Observation:'. The system will execute the tool and provide the observation back to you.
6. ALL your internal thoughts and tool actions MUST be in English. ONLY the Final Answer should be translated to {state.get("language", "English")}.
7. RETRY AND REFORMULATE ON FAILURE: If a search or retrieve tool returns no results, irrelevant context, or a tool error, do not give up immediately. Reflect on why it failed, reformulate your query (use synonyms, translation, broader/narrower search terms), and execute the tool again with the updated query.
8. WEB SEARCH ON LOW RELEVANCE: If the retrieve_tool warns that the maximum relevance score of the retrieved chunks is low (e.g., < 0.5), it means the local document does not contain the answer. In this case, you MUST immediately call the web_search_tool with a well-formulated search query to find the answer on the web instead of retrieving from the database again.
9. MANDATORY THINKING PROCESS: Under each 'Thought:' section, you must write a detailed explanation (1-2 sentences) of your reasoning process before specifying the Action or Final Answer. Do not jump directly to 'Action:' or leave the thought empty.

Use the following strict format:

Question: the input question you must answer
Thought: you should always think about what to do next
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, INCLUDING NUMERICAL INDEX CITATIONS (e.g., [1], [2]).

--- Example of a valid thought process ---
Question: What is the Transformer architecture?
Thought: I need to search the database for information about the Transformer architecture.
Action: retrieve_tool
Action Input: Transformer architecture overview
Observation: --- Document [1] (Score: 0.85, Source: attention_is_all_you_need.pdf) ---
[SECTION]
Introduction

[CONTENT]
The Transformer is a model architecture eschewing recurrence and instead relying entirely on an attention mechanism...
Thought: I have enough information to answer the question.
Final Answer: The Transformer architecture relies entirely on an attention mechanism, eschewing recurrence [1].
------------------------------------------

Begin!

Current Loop Iteration: {iterations + 1} of 4.{loop_warning}

Question: {query}
{scratchpad}"""

        logger.info(f"[{self.name}] Reasoning Node Iteration {iterations}")
        
        try:
            # Call Ollama with stop words to prevent hallucinating the observation
            response_text = self.llm.generate(prompt, stop=["\nObservation:", "Observation:"])
            
            # ── Guard: if model returns empty output, retry once with a nudge ──
            if not response_text or not response_text.strip():
                logger.warning(f"[{self.name}] LLM returned empty response. Retrying with explicit nudge.")
                nudge_prompt = prompt + " I need to use the retrieve_tool to find relevant information."
                response_text = self.llm.generate(nudge_prompt, stop=["\nObservation:", "Observation:"])
                
            if not response_text or not response_text.strip():
                logger.error(f"[{self.name}] LLM returned empty response twice. Forcing fallback.")
                return {
                    "final_answer": "Xin lỗi, tôi không thể xử lý yêu cầu này. Mô hình ngôn ngữ không tạo ra phản hồi.",
                    "iterations": iterations + 1
                }
            
            # Explicitly log the LLM's raw output for full visibility
            logger.info(f"[{self.name}] Raw Output:\n{response_text}")
            
            # Parse output using regex
            # Look for Final Answer
            final_answer_match = re.search(r"Final Answer:(.*?)(?:$|\n)", response_text, re.DOTALL)
            if final_answer_match:
                final_answer = final_answer_match.group(1).strip()
                logger.info(f"[{self.name}] Found Final Answer: {final_answer}")
                return {"final_answer": final_answer, "iterations": iterations + 1}
                
            # Look for Action and Action Input
            action_match = re.search(r"Action:\s*(.*?)\n", response_text)
            action_input_match = re.search(r"Action Input:\s*(.*?)(?:\n|$)", response_text)
            
            # Extract Thought before the Action
            thought_match = re.search(r"Thought:\s*(.*?)\nAction:", response_text, re.DOTALL)
            if not thought_match:
                thought_match = re.search(r"^(.*?)\nAction:", response_text, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else response_text.strip()
            
            # Save thought to explicitly log in structured output
            thoughts = state.get("thought_process", [])
            thoughts.append(thought)
            logger.info(f"[{self.name}] Parsed Thought: {thought}")
            
            new_scratchpad = scratchpad + f" {response_text}\n"
            
            if action_match and action_input_match:
                action = action_match.group(1).strip()
                action_input = action_input_match.group(1).strip()
                
                logger.info(f"[{self.name}] Parsed Action: {action} with Input: {action_input}")
                
                return {
                    "scratchpad": new_scratchpad,
                    "thought_process": thoughts,
                    "pending_action": {"tool": action, "input": action_input},
                    "iterations": iterations + 1
                }
            
            # If no action and no final answer, force a final answer to prevent infinite loops
            logger.warning(f"[{self.name}] Failed to parse Action or Final Answer. Forcing end.")
            return {"final_answer": response_text.strip(), "iterations": iterations + 1}
            
        except Exception as e:
            logger.error(f"[{self.name}] Reasoning failed: {e}")
            return {"final_answer": f"Error during reasoning: {str(e)}", "iterations": iterations + 1}

    def _tool_node(self, state: AgentGraphState) -> dict:
        pending_action = state.get("pending_action")
        if not pending_action:
            return {}
            
        raw_action_name = pending_action["tool"]
        action_input = pending_action["input"]
        
        # Soft-match for local LLM hallucinations, but prevent matching full sentences
        action_name = raw_action_name
        name_lower = raw_action_name.lower().replace(" ", "_").replace("-", "_")
        
        if len(name_lower) < 40: # Prevent mapping entire hallucinated sentences
            if "retrieve" in name_lower or "database" in name_lower:
                action_name = "retrieve_tool"
            elif "search" in name_lower or "web" in name_lower:
                action_name = "web_search_tool"
            elif "vision" in name_lower or "image" in name_lower:
                action_name = "vision_tool"
        
        logger.info(f"[{self.name}] Executing Tool: {action_name} (Raw: '{raw_action_name}') with input: {action_input}")
        
        if action_name in self.tool_map:
            try:
                # Call tool
                tool_func = self.tool_map[action_name]
                observation = tool_func.invoke(action_input)
            except Exception as e:
                observation = f"Tool Error: {str(e)}"
        else:
            tool_list = ", ".join(self.tool_map.keys())
            observation = f"Error: Tool '{raw_action_name}' is invalid. You MUST use EXACTLY one of these tools: [{tool_list}]. Do NOT write sentences in the Action field."
            
        logger.info(f"[{self.name}] Tool Observation (length {len(str(observation))})")
        
        # Dynamic index correction for multiple retrieve calls
        if action_name == "retrieve_tool" and isinstance(observation, str):
            existing_count = len(state.get("retrieved_docs", []))
            pattern = r'--- Document \[(\d+)\]'
            
            counter = [existing_count]
            def replace_index(match):
                counter[0] += 1
                return f"--- Document [{counter[0]}]"
                
            observation = re.sub(pattern, replace_index, observation)

        retrieved_docs = state.get("retrieved_docs", [])
        if action_name == "retrieve_tool" and isinstance(observation, str):
            chunks = re.split(r"--- Document \[\d+\].*?---", observation)
            for c in chunks:
                c_clean = c.strip()
                if c_clean:
                    content_match = re.search(r"\[CONTENT\]\n(.*)", c_clean, re.DOTALL)
                    if content_match:
                        retrieved_docs.append(content_match.group(1).strip())
                    else:
                        retrieved_docs.append(c_clean)
        # Truncate the observation written to the scratchpad to save context window space
        obs_str = str(observation)
        if len(obs_str) > 30000:
            obs_str = obs_str[:30000] + f"\n\n... [TRUNCATED due to length limits (original size: {len(obs_str)}) to prevent context window collapse] ..."

        # Append observation to scratchpad
        scratchpad = state.get("scratchpad", "")
        new_scratchpad = scratchpad + f"Observation: {obs_str}\n"
        
        context_used = state.get("context_used", [])
        context_used.append(f"Used {action_name}. Retrieved {len(str(observation))} characters.")
            
        return {
            "scratchpad": new_scratchpad,
            "context_used": context_used,
            "retrieved_docs": retrieved_docs,
            "pending_action": None # Clear pending action
        }

    def _route_after_reasoning(self, state: AgentGraphState) -> str:
        if state.get("final_answer"):
            return "end"
        if state.get("iterations", 0) > 3:
            logger.warning(f"[{self.name}] Max iterations reached. Terminating sub-graph.")
            return "end"
        return "tool"

    def run(self, query: str, language: str = "English") -> AgentResponse:
        logger.info(f"=== Starting {self.name} Sub-Graph ===")
        initial_state = AgentGraphState(
            query=query,
            language=language,
            scratchpad="",
            thought_process=[],
            context_used=[],
            pending_action=None,
            final_answer=None,
            iterations=0,
            retrieved_docs=[]
        )
        
        final_state = self.app.invoke(initial_state)
        
        thought_process_str = "\n---\n".join(final_state.get("thought_process", []))
        
        logger.info(f"=== {self.name} Sub-Graph Finished ===")
        return AgentResponse(
            thought_process=thought_process_str,
            content=final_state.get("final_answer") or "No final answer.",
            context_used=final_state.get("context_used", []),
            retrieved_docs=final_state.get("retrieved_docs", [])
        )
