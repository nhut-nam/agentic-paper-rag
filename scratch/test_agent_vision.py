import os
import sys
from typing import List, Dict, Any

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw
from app.utils.db import DatabaseHandler
from app.utils.storage import StorageManager
from app.llm.base import BaseLLMProvider
from app.agent.research_agent import ResearchAgent

class MockLLM(BaseLLMProvider):
    """
    Mock LLM provider to simulate the step-by-step reasoning process of the ResearchAgent.
    This allows the test to run and demonstrate the logic without requiring an active local Ollama instance.
    """
    def __init__(self):
        self.call_count = 0

    def generate(self, prompt: str, stop: List[str] = None) -> str:
        self.call_count += 1
        
        # Step 1: Agent decides to search the database first
        if self.call_count == 1:
            print("\n[Mock LLM] Step 1: LLM decides to retrieve document details...")
            return (
                "Thought: I need to search the database to get information about the Figure 1 structure.\n"
                "Action: retrieve_tool\n"
                "Action Input: Figure 1 structure layout\n"
            )
            
        # Step 2: Agent receives the chunk containing the relative image path '../images/fig_model_arch/image.png'
        # and decides to invoke the vision_tool
        elif self.call_count == 2:
            print("\n[Mock LLM] Step 2: LLM detects image path '../images/fig_model_arch/image.png' in retrieve results and decides to call vision_tool...")
            return (
                "Thought: The retrieved document content mentions a layout diagram at '../images/fig_model_arch/image.png'. "
                "I must analyze this image directly to see what specific components are shown in the Figure 1 diagram.\n"
                "Action: vision_tool\n"
                "Action Input: ../images/fig_model_arch/image.png\n"
            )
            
        # Step 3: Agent receives the vision analysis description, synthesizes the results, and returns the final answer
        else:
            print("\n[Mock LLM] Step 3: LLM compiles retrieve details & vision analysis into the final answer...")
            return (
                "Thought: I have the description of the components from the document text and the visual title "
                "from Figure 1 block diagram ('Figure 1: Transformer Block Overview'). I can now write the final response.\n"
                "Final Answer: According to the document, the components include Multi-Head Attention, Feed Forward networks, "
                "and residual connections. The Figure 1 diagram visually represents a 'Transformer Block Overview' "
                "[Source: test_vision_doc].\n"
            )

    def generate_summary_and_keywords(self, content: str) -> Dict[str, Any]:
        return {"summary": "Mock summary", "keywords": ["mock"]}

    def get_embeddings(self, text: str) -> List[float]:
        # 384 dimensional dummy vector
        return [0.0] * 384

def setup_dummy_data():
    """
    Sets up a dummy document and a chunk containing an image path in the database.
    Also creates a dummy image on disk to show how the path is resolved.
    """
    db = DatabaseHandler()
    storage = StorageManager()
    
    doc_id = "test_vision_doc"
    doc_path = "storage/uploads/test_vision_doc.pdf"
    
    print("1. Inserting dummy document into database...")
    db.insert_document(doc_id, doc_path)
    db.update_status(doc_id, "chunked")
    
    print("2. Creating a dummy image on disk...")
    # Target path for resolver: storage/processed/{doc_id}/images/fig_model_arch/image.png
    relative_image_dir = "processed/test_vision_doc/images/fig_model_arch"
    storage.ensure_dir(relative_image_dir)
    image_full_path = storage.get_full_path(f"{relative_image_dir}/image.png")
    
    # Generate simple image using PIL
    img = Image.new('RGB', (350, 100), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10, 40), "Figure 1: Transformer Block Overview", fill=(255, 255, 0))
    img.save(image_full_path)
    print(f"   Created dummy image at: {image_full_path}")
    
    print("3. Inserting dummy chunk into database...")
    chunk_content = (
        "We present the detailed neural network architecture in Figure 1. "
        "The components include Multi-Head Attention, Feed Forward networks, and residual connections. "
        "Refer to the layout diagram here: ../images/fig_model_arch/image.png for structural flows."
    )
    
    chunk_data = {
        "chunk_id": "test_vision_ch_1",
        "doc_id": doc_id,
        "content": chunk_content,
        "heading_path": "Architecture > Model Structure",
        "section_title": "Figure 1 Layout",
        "chunk_order": 0,
        "summary": "This chunk describes Figure 1 which shows the detailed neural network architecture layout of the model, located at ../images/fig_model_arch/image.png.",
        "keywords": ["figure 1", "architecture", "diagram", "model structure", "fig_model_arch"],
        "embedding": [0.01] * 384 
    }
    
    db.insert_chunk(chunk_data)
    print("   Inserted dummy chunk into PostgreSQL.")
    return doc_id

def run_test(doc_id: str):
    """
    Runs the ResearchAgent with the mock LLM, demonstrating the tool orchestration.
    """
    print("\n4. Initializing Mock LLM & ResearchAgent...")
    mock_llm = MockLLM()
    
    # Initialize ResearchAgent scoped to our dummy doc_id
    agent = ResearchAgent(llm=mock_llm, doc_id=doc_id)
    
    query = "Based on the document, what specific components are shown in the Figure 1 architecture diagram?"
    
    print(f"\n5. Sending query to ResearchAgent: '{query}'")
    print("=" * 60)
    
    # Execute the Agent graph
    response = agent.run(query, language="English", mode="answer")
    
    print("=" * 60)
    print("\n--- AGENT THOUGHT PROCESS (LOGGED STEPS) ---")
    print(response.thought_process)
    print("\n--- FINAL ANSWER FROM AGENT ---")
    print(response.content)
    print("\n--- CONTEXT RETRIEVED BY AGENT ---")
    for ctx in response.context_used:
        print(f"- {ctx}")

def cleanup_dummy_data(doc_id: str):
    """Clean up DB records and files created for the test."""
    print("\n6. Cleaning up database records and dummy files...")
    db = DatabaseHandler()
    try:
        db.delete_document(doc_id)
        print("   Database records cleared.")
    except Exception as e:
        print(f"   Failed to delete database records: {e}")
        
    storage = StorageManager()
    relative_image_dir = "processed/test_vision_doc"
    try:
        storage.clear_dir(relative_image_dir)
        os.rmdir(storage.get_full_path(relative_image_dir))
        print("   Dummy image files deleted.")
    except Exception as e:
        print(f"   Failed to delete dummy files: {e}")

if __name__ == "__main__":
    doc_id = None
    try:
        doc_id = setup_dummy_data()
        run_test(doc_id)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
    finally:
        if doc_id:
            cleanup_dummy_data(doc_id)
