from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from app.models.schemas import Plan, Task, AgentType
from app.agent.analyzer import ComplexityAnalyzer
from app.agent.planner import Planner
from app.agent.research_agent import ResearchAgent
from app.agent.general_agent import GeneralAgent
from app.agent.synthesizer_agent import SynthesizerAgent
from app.utils.logger import logger
from app.llm.base import BaseLLMProvider
from app.utils.db import DatabaseHandler

# 1. Define the State Schema
class AgentState(TypedDict):
    query: str
    query_en: Optional[str]
    doc_id: Optional[str]
    language: Optional[str]
    mode: Optional[str]
    global_metadata: str
    agent_metadata: str
    plan: Optional[Plan]
    tasks: List[Task]
    current_task_idx: int
    global_context: List[str]
    final_answer: Optional[str]
    memory_id: Optional[int]
    retrieved_docs: List[str]
    session_id: Optional[str]
    session_summary: Optional[str]
    is_ambiguous: Optional[bool]

class AgentWorkflow:
    """
    Encapsulates the LangGraph execution loop for the Agent System using Plan-and-Execute.
    """
    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm_provider = llm_provider
        self.analyzer = ComplexityAnalyzer(llm=self.llm_provider)
        self.planner = Planner(llm=self.llm_provider)
        self.synthesizer = SynthesizerAgent(llm=self.llm_provider)
        self.db = DatabaseHandler()
        self.app = self._build_graph()

    def _analyzer_node(self, state: AgentState):
        logger.info("--- ENTERING ANALYZER NODE ---")
        query = state["query"]
        session_summary = state.get("session_summary")
        analysis = self.analyzer.analyze(query, session_summary=session_summary)
        
        # Use rewritten query if analyzer refined it
        rewritten = analysis.rewritten_query or query
        if rewritten != query:
            logger.info(f"Analyzer refined query from '{query}' to '{rewritten}' using session summary.")
            
        # Translate to English if not English to avoid local LLM parsing/language confusion
        query_en = rewritten
        if analysis.language and analysis.language.lower() != "english":
            prompt = f"""You are a translator. Translate the following user query to English. 
            Do NOT include any preamble, introduction, explanation, or conversational text. 
            Translate the text literally and directly.
            
            Text to translate:
            "{rewritten}"
            
            English Translation:"""
            try:
                translated = self.llm_provider.generate(prompt).strip()
                if translated:
                    if translated.startswith('"') and translated.endswith('"'):
                        translated = translated[1:-1]
                    query_en = translated
                    logger.info(f"Translated query from {analysis.language} to English: '{query_en}'")
            except Exception as e:
                logger.error(f"Failed to translate query to English: {e}")
        
        return {
            "mode": analysis.mode,
            "language": analysis.language,
            "query_en": query_en,
            "query": rewritten,
            "is_ambiguous": analysis.is_ambiguous
        }

    def _planner_node(self, state: AgentState):
        logger.info("--- ENTERING PLANNER NODE ---")
        query_en = state.get("query_en") or state["query"]
        mode = state.get("mode", "answer")
        global_metadata = state.get("global_metadata", "")
        
        # Define agent capabilities for the planner
        agent_metadata = """
        [
            {"name": "research", "description": "Expert researcher equipped with tools to retrieve document context from the vector DB, search the web, and use vision to analyze images. Use this agent for finding information and answering questions based on context."}
        ]
        """
        
        plan, tasks = self.planner.plan(
            query=query_en, 
            language="English", 
            mode=mode, 
            global_metadata=global_metadata, 
            agent_metadata=agent_metadata
        )
        
        return {
            "plan": plan,
            "tasks": tasks,
            "current_task_idx": 0,
            "agent_metadata": agent_metadata
        }

    def _executor_node(self, state: AgentState):
        logger.info("--- ENTERING EXECUTOR NODE ---")
        tasks = state.get("tasks", [])
        idx = state.get("current_task_idx", 0)
        
        if idx >= len(tasks):
            return {} # Should not happen if routing is correct
            
        current_task = tasks[idx]
        logger.info(f"Executing Task {idx+1}/{len(tasks)}: [{current_task.agent_type.value.upper()}] {current_task.content}")
        
        retrieved = []
        try:
            # Route to the appropriate agent
            if current_task.agent_type == AgentType.RESEARCH:
                # Initialize agent dynamically with scoped doc_id
                doc_id = state.get("doc_id")
                research_agent = ResearchAgent(llm=self.llm_provider, doc_id=doc_id)
                response_obj = research_agent.run(current_task.content, language="English", mode=state.get("mode", "answer"))
                result_text = response_obj.content
                thought_process = response_obj.thought_process
                retrieved = response_obj.retrieved_docs
            else:
                logger.warning(f"Agent {current_task.agent_type} not fully implemented yet. Falling back to ResearchAgent.")
                doc_id = state.get("doc_id")
                research_agent = ResearchAgent(llm=self.llm_provider, doc_id=doc_id)
                response_obj = research_agent.run(current_task.content, language="English", mode=state.get("mode", "answer"))
                result_text = response_obj.content
                thought_process = response_obj.thought_process
                retrieved = response_obj.retrieved_docs
                
            # Format context addition
            context_entry = f"Task: {current_task.content}\nResult: {result_text}"
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}", exc_info=True)
            context_entry = f"Task: {current_task.content}\nResult: Error encountered - {e}"
            
        current_context = state.get("global_context", [])
        current_context.append(context_entry)
        
        current_retrieved_docs = state.get("retrieved_docs", [])
        current_retrieved_docs.extend(retrieved)
        
        return {
            "global_context": current_context,
            "retrieved_docs": current_retrieved_docs,
            "current_task_idx": idx + 1
        }

    def _synthesizer_node(self, state: AgentState):
        logger.info("--- ENTERING SYNTHESIZER NODE ---")
        query = state["query"]
        language = state.get("language", "English")
        mode = state.get("mode", "answer")
        context_list = state.get("global_context", [])
        retrieved_docs = state.get("retrieved_docs", [])
        
        combined_context = "\n\n---\n\n".join(context_list)
        
        try:
            response_obj = self.synthesizer.run(
                query=query,
                language=language,
                mode=mode,
                findings=combined_context,
                retrieved_docs=retrieved_docs
            )
            final_answer = response_obj.content
        except Exception as e:
            logger.error(f"SynthesizerAgent failed: {e}", exc_info=True)
            final_answer = "Sorry, an error occurred while synthesizing the final answer."
            
        return {
            "final_answer": final_answer
        }

    def _general_agent_node(self, state: AgentState):
        logger.info("--- ENTERING GENERAL AGENT NODE ---")
        query = state["query"]
        language = state.get("language", "English")
        is_ambiguous = state.get("is_ambiguous", False)
        
        general_agent = GeneralAgent(llm=self.llm_provider)
        response_obj = general_agent.run(query, language=language, is_ambiguous=is_ambiguous)
        
        return {
            "final_answer": response_obj.content
        }

    def _route_after_analyzer(self, state: AgentState) -> str:
        mode = state.get("mode")
        if mode == "general" or state.get("is_ambiguous"):
            return "general_agent"
        return "planner"

    def _route_execution(self, state: AgentState) -> str:
        """
        Routes either to the next task in the executor, or to the synthesizer if all tasks are done.
        """
        tasks = state.get("tasks", [])
        idx = state.get("current_task_idx", 0)
        
        if idx < len(tasks):
            return "execute"
        else:
            return "synthesize"

    def _build_graph(self):
        """
        Constructs and compiles the LangGraph StateGraph for Plan-and-Execute.
        """
        workflow = StateGraph(AgentState)

        workflow.add_node("analyzer", self._analyzer_node)
        workflow.add_node("general_agent", self._general_agent_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("synthesizer", self._synthesizer_node)

        # Flow
        workflow.set_entry_point("analyzer")
        
        # Branch after analyzer
        workflow.add_conditional_edges(
            "analyzer",
            self._route_after_analyzer,
            {
                "general_agent": "general_agent",
                "planner": "planner"
            }
        )
        
        # General agent bypasses planning
        workflow.add_edge("general_agent", END)
        
        # From planner, we check if there are tasks to execute
        workflow.add_conditional_edges(
            "planner",
            self._route_execution,
            {
                "execute": "executor",
                "synthesize": "synthesizer"
            }
        )
        
        # After execution, we check again
        workflow.add_conditional_edges(
            "executor",
            self._route_execution,
            {
                "execute": "executor",
                "synthesize": "synthesizer"
            }
        )
        
        workflow.add_edge("synthesizer", END)

        return workflow.compile()

    def run(self, query: str, global_metadata: str = "", doc_id: str = None, session_id: str = None) -> AgentState:
        """
        Executes the LangGraph workflow for a given query, supporting session context and memory.
        """
        import uuid
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Generated new session_id: {session_id}")
            
        # 1. Ensure the session exists in the database
        try:
            self.db.create_session_chat(session_id)
            if doc_id:
                self.db.link_session_document(session_id, doc_id)
        except Exception as e:
            logger.error(f"Failed to initialize session in database: {e}")

        # 2. Retrieve the latest summary for this session
        session_summary = None
        try:
            summary_record = self.db.get_latest_summary_session(session_id)
            if summary_record:
                session_summary = summary_record["summary"]
                logger.info(f"Loaded existing session summary for session: {session_id}")
        except Exception as e:
            logger.error(f"Failed to load latest session summary: {e}")

        logger.info(f"Starting workflow for query: '{query}' (doc_id: {doc_id}, session_id: {session_id})")
        initial_state = AgentState(
            query=query,
            query_en=None,
            doc_id=doc_id,
            language=None,
            mode=None,
            global_metadata=global_metadata,
            agent_metadata="",
            plan=None,
            tasks=[],
            current_task_idx=0,
            global_context=[],
            final_answer=None,
            memory_id=None,
            retrieved_docs=[],
            session_id=session_id,
            session_summary=session_summary
        )
        
        final_state = self.app.invoke(initial_state)
        logger.info("--- WORKFLOW COMPLETE ---")
        
        final_answer = final_state.get("final_answer", "No answer generated.")
        logger.info(f"=== FINAL ANSWER ===\n{final_answer}\n====================")
        
        # Save query interaction to database memory for Ragas evaluation
        memory_id = None
        try:
            contexts = final_state.get("retrieved_docs", [])
            if not contexts:
                contexts = final_state.get("global_context", [])
            if not contexts:
                contexts = ["No context retrieved."]
            memory_id = self.db.insert_query_memory(
                query=query, # Save original user query
                response=final_answer,
                retrieved_contexts=contexts,
                doc_id=doc_id,
                session_id=session_id
            )
            
            # Generate updated session summary using LLM
            if final_state.get("mode") != "general" or session_summary:
                prompt = f"""
                You are an AI Conversation Summarizer. Your job is to update the summary of a Q&A conversation.
                
                Previous Conversation Summary:
                "{session_summary or 'No prior conversation.'}"
                
                Latest Turn:
                User Query: "{final_state.get('query')}"
                Assistant Response: "{final_answer}"
                
                Provide an updated, concise, high-level summary of the key information discussed so far in this conversation. 
                Keep it in the language of the conversation (preferably Vietnamese or English as used). 
                Do NOT use dialogue or conversational preamble. Write a single dense paragraph.
                
                Updated Summary:"""
                
                try:
                    new_summary = self.llm_provider.generate(prompt).strip()
                    if new_summary:
                        self.db.insert_summary_session(
                            session_id=session_id,
                            summary=new_summary,
                            memory_id=memory_id
                        )
                        logger.info(f"Updated session summary saved to database.")
                except Exception as e:
                    logger.error(f"Failed to generate/save updated session summary: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to log interaction to query memory: {e}", exc_info=True)
            
        final_state["memory_id"] = memory_id
        final_state["session_id"] = session_id
        return final_state
