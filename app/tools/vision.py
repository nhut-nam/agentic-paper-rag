import os
from langchain_core.tools import tool
from typing import Optional, List
from app.utils.logger import logger
import ollama

def get_vision_tool(doc_id: Optional[str] = None, doc_ids: Optional[List[str]] = None):
    """
    Returns a configured vision_tool scoped to a specific doc_id or list of doc_ids to resolve relative image paths.
    """
    @tool("vision_tool")
    def vision_tool(image_path: str) -> str:
        """
        Analyzes an image and returns a detailed description. 
        Use this tool to extract insights from visual content like charts, graphs, or diagrams when you encounter image links in the document.
        
        Args:
            image_path (str): The path to the image file (e.g. '../images/{image_id}/image.png').
        """
        logger.info(f"Vision tool invoked with image_path: '{image_path}', doc_id: '{doc_id}', doc_ids: '{doc_ids}'")
        
        # Target ids list
        target_ids = []
        if doc_ids:
            target_ids.extend(doc_ids)
        if doc_id and doc_id not in target_ids:
            target_ids.append(doc_id)

        # 1. Resolve relative path to actual disk path
        resolved_path = image_path
        
        if image_path.startswith("../images/") and target_ids:
            # "../images/fig1/image.png" -> "storage/processed/{doc_id}/images/fig1/image.png"
            relative_part = image_path.replace("../images/", "")
            for t_id in target_ids:
                temp_path = f"storage/processed/{t_id}/images/{relative_part}"
                if os.path.exists(temp_path):
                    resolved_path = temp_path
                    break
            else:
                resolved_path = f"storage/processed/{target_ids[0]}/images/{relative_part}"
                
        elif "images/" in image_path and target_ids:
            # Handle cases where LLM might omit "../"
            # e.g., "images/fig1/image.png" -> "storage/processed/{doc_id}/images/fig1/image.png"
            if not image_path.startswith("storage/"):
                parts = image_path.split("images/")
                relative_part = parts[-1]
                for t_id in target_ids:
                    temp_path = f"storage/processed/{t_id}/images/{relative_part}"
                    if os.path.exists(temp_path):
                        resolved_path = temp_path
                        break
                else:
                    resolved_path = f"storage/processed/{target_ids[0]}/images/{relative_part}"
                
        # 2. Try secondary fallback lookup using storage base dir if still not found
        if not os.path.exists(resolved_path):
            try_path = os.path.join("storage", image_path.replace("../", ""))
            if os.path.exists(try_path):
                resolved_path = try_path
        
        logger.info(f"Vision tool resolved image path: '{image_path}' -> '{resolved_path}'")
        
        if not os.path.exists(resolved_path):
            logger.error(f"Image not found at resolved path: {resolved_path}")
            return f"Error: Image not found at path {image_path}. Please check if the path is correct."
            
        try:
            logger.info(f"Analyzing image {resolved_path} with minicpm-v...")
            response = ollama.chat(
                model='minicpm-v',
                messages=[{
                    'role': 'user',
                    'content': 'Describe this image in detail. Extract all text, mathematical formulas, and parameters exactly as they appear in the image without omitting anything.',
                    'images': [resolved_path]
                }],
                options={
                    "num_predict": 1024,
                    "temperature": 0.0
                }
            )
            description = response.get('message', {}).get('content', '')
            return f"--- Image Analysis Result ---\nSource: {image_path}\nDescription:\n{description}"
        except Exception as e:
            logger.error(f"Vision tool failed: {e}")
            return f"Error executing image analysis: {str(e)}"

    return vision_tool

# Default instance for backward compatibility
vision_tool = get_vision_tool()

