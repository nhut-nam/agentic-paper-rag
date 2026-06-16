from typing import List, Dict, Any
from app.utils.logger import logger
from app.utils.db import DatabaseHandler
from app.utils.storage import StorageManager
from app.utils.context_manager import ContextManager
from app.llm.factory import LLMFactory

class PaperAnalyzer:
    """
    Service responsible for generating a global, pre-computed analysis of a paper 
    using a Map-Reduce approach over its chunk summaries.
    """

    def __init__(self):
        self.db = DatabaseHandler()
        self.storage = StorageManager()
        self.context_manager = ContextManager()
        self.llm = LLMFactory.get_provider("ollama")
        
        # Batch size for Map-Reduce. 
        # A batch of 10 chunks (where each summary is ~50-100 words) = ~500-1000 words.
        # This easily fits in a standard context window and runs fast.
        self.batch_size = 10 

    def _reduce_func(self, batch: List[str]) -> str:
        """
        The Map-Reduce worker function. Takes a batch of formatted chunk summaries 
        and condenses them into a single coherent section summary.
        """
        combined_text = "\n\n".join(batch)
        
        prompt = f"""
        You are an expert academic research assistant.
        Below are sequential summaries of several sections from a research paper.
        
        {combined_text}
        
        Your task:
        Write a comprehensive but concise summary of this entire section. 
        Ensure you capture the logical flow, key methodologies, and any critical findings mentioned.
        Do not add external information.
        
        Summary:
        """
        
        response = self.llm.generate(prompt)
        return response.strip()

    def generate_analysis(self, doc_id: str) -> bool:
        """
        Main entry point. Fetches chunks, performs Map-Reduce, and generates the final analysis.md.
        """
        logger.info(f"PaperAnalyzer: Starting analysis for {doc_id}")
        
        chunks = self.db.get_chunks_by_doc_id(doc_id)
        if not chunks:
            logger.error(f"PaperAnalyzer: No chunks found for {doc_id}")
            return False

        # 1. Format chunks for ContextManager
        formatted_chunks = []
        for chunk in chunks:
            formatted_chunks.append(
                f"[{chunk.heading_path}] {chunk.section_title}:\n{chunk.summary}"
            )

        # 2. Map-Reduce all chunks into a single Global Summary
        logger.info(f"PaperAnalyzer: Running Map-Reduce on {len(formatted_chunks)} chunks...")
        global_summary = self.context_manager.map_reduce(
            items=formatted_chunks,
            batch_size=self.batch_size,
            reduce_func=self._reduce_func
        )

        # 3. Final Synthesis (Generate structured analysis.md)
        logger.info(f"PaperAnalyzer: Generating final structured report...")
        final_prompt = f"""
        You are an expert academic research assistant.
        Based on the following global summary of a research paper, generate a structured Executive Analysis in Markdown format.
        
        Global Summary:
        {global_summary}
        
        You MUST strictly follow this Markdown structure in English:
        # Abstract
        [Provide a high-level summary of what the paper is about]
        
        # Objectives
        [What problems is the paper trying to solve?]
        
        # Main Concepts
        [What are the core technical concepts, models, or frameworks introduced and focused on in this paper?]
        
        # Related Concepts
        [What are the related works, prior methods, or broader field concepts mentioned for context?]
        
        # Methodology
        [How did the authors solve the problem? What models/algorithms/data did they use?]
        
        # Key Findings
        [What were the main results and achievements?]
        
        # Limitations
        [What are the weaknesses or explicitly stated limitations of the approach?]
        
        Ensure the output is entirely in English. Keep it professional, concise, and academic.
        """
        
        final_markdown = self.llm.generate(final_prompt)
        
        if not final_markdown:
            logger.error("PaperAnalyzer: Failed to generate final markdown.")
            return False

        # 4. Save to Storage
        output_path = f"processed/{doc_id}/analysis.md"
        self.storage.save_text(output_path, final_markdown.strip())
        logger.info(f"PaperAnalyzer: Successfully saved analysis to storage/{output_path}")
        
        return True
