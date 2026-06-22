import re
import os
from langchain_core.tools import tool
from typing import Optional, List
from app.pipelines.retrieve import RetrievePipeline
from app.utils.logger import logger
from app.utils.db import DatabaseHandler

def get_retrieve_tool(doc_id: Optional[str] = None, doc_ids: Optional[List[str]] = None):
    """
    Returns a configured retrieve_tool that is scoped to a specific doc_id or list of doc_ids if provided.
    """
    @tool("retrieve_tool")
    def retrieve_tool(query: str, top_k: int = 3) -> str:
        """
        Search the local knowledge base (database) for documents matching the given query.
        Use this tool whenever you need to find factual information, context, or documents related to the user's topic.
        Returns the retrieved content as a formatted string with structured sections, contents, and related images/tables/equations.
        
        Args:
            query (str): The search query to look for in the database.
            top_k (int, optional): The maximum number of results to return. Defaults to 3.
        """
        logger.info(f"Retrieve tool invoked with query: '{query}', top_k: {top_k}, doc_id: {doc_id}, doc_ids: {doc_ids}")
    
        try:
            # Initialize the existing pipeline
            pipeline = RetrievePipeline()
            results = pipeline.run(query=query, top_k=top_k, doc_id=doc_id, doc_ids=doc_ids)
            
            if not results:
                return f"No relevant documents found for query: '{query}'."
                
            db = DatabaseHandler()
            formatted_results = []
            for i, res in enumerate(results):
                chunk = res.get("chunk")
                score = res.get("combined_score", 0.0)
                
                if not chunk:
                    continue
                    
                # Safely get attributes
                doc_name = getattr(chunk, 'doc_id', 'Unknown Source')
                content = getattr(chunk, 'content', '')
                
                heading_path = getattr(chunk, 'heading_path', '')
                if not heading_path or heading_path == 'Root':
                    heading_path = getattr(chunk, 'section_title', 'Root')
                
                # 1. Scan for image ids in the chunk's content
                # Matches pattern: ../images/{img_id}/image.png
                image_ids = re.findall(r'\.\./images/([^/]+)/image\.png', content)
                
                related_images_text = []
                if image_ids:
                    # 2. Fetch metadata for these images from database
                    images_metadata = db.get_images_by_ids(image_ids)
                    for img in images_metadata:
                        img_type = img.get("image_type", "Figure").upper()
                        img_id = img.get("image_id", "")
                        img_content = img.get("content", "").strip()
                        img_rel_path = img.get("image_path", "")
                        
                        # Generate absolute system path for the image file
                        img_abs_path = os.path.abspath(os.path.join("storage", img_rel_path))
                        
                        if img_content:
                            related_images_text.append(
                                f"[RELATED {img_type}]\n"
                                f"Image ID: {img_id}\n"
                                f"{img_content}"
                            )
                        else:
                            # If no text could be extracted, provide file path for vision tool
                            related_images_text.append(
                                f"[RELATED {img_type}]\n"
                                f"Image ID: {img_id}\n"
                                f"No text content could be extracted. Use vision tool to analyze the image file directly if needed.\n"
                                f"Image File Path: {img_abs_path}"
                            )
                
                # 3. Format result block
                chunk_formatted = (
                    f"--- Document [{i+1}] (Score: {score:.2f}, Source: {doc_name}) ---\n"
                    f"[SECTION]\n{heading_path}\n\n"
                    f"[CONTENT]\n{content}\n"
                )
                
                if related_images_text:
                    chunk_formatted += "\n" + "\n\n".join(related_images_text) + "\n"
                    
                formatted_results.append(chunk_formatted)
                
            # Calculate maximum score
            max_score = max([res.get("combined_score", 0.0) for res in results]) if results else 0.0
            
            output_text = "\n\n".join(formatted_results)
            if max_score < 0.5:
                warning_text = (
                    f"\n\n[WARNING] The maximum relevance score of the local database chunks is very low ({max_score:.2f} < 0.5).\n"
                    "This strongly suggests that the local paper/documents do not contain the answer to your query.\n"
                    "To find the correct answer, you MUST now use the web_search_tool instead of trying to retrieve from the database again.\n"
                )
                output_text += warning_text
                logger.info(f"Retrieve score is low ({max_score:.2f} < 0.5). Appended warning message to tool output.")
            
            return output_text
            
        except Exception as e:
            logger.error(f"Retrieve tool failed: {e}")
            return f"Error executing retrieval: {str(e)}"
            
    return retrieve_tool
    
# Provide a default instance for backward compatibility
retrieve_tool = get_retrieve_tool()

