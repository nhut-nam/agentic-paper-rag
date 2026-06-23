import os
import sys
from dotenv import load_dotenv

# Reconfigure stdout to use UTF-8 on Windows
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.agent_loop import AgentWorkflow
from app.llm.factory import LLMFactory

def test_workflow():
    print("============================================================")
    print(" VERIFIER LOOP-BACK INTEGRATION TEST")
    print("============================================================")
    
    print("1. Initializing Ollama LLM provider...")
    try:
        llm_provider = LLMFactory.get_provider("ollama")
        print("   Ollama LLM provider initialized successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to get LLM provider: {e}")
        return
        
    print("\n2. Initializing AgentWorkflow...")
    workflow = AgentWorkflow(llm_provider=llm_provider)
    print("   AgentWorkflow compiled and ready.")
    
    # A query requesting specific hyperparameters and results
    query = "Theo tài liệu, các tác giả đã sử dụng thuật toán tối ưu hóa (optimizer) nào cùng với batch size bao nhiêu cho quá trình tiền huấn luyện (pre-training) trên tập JFT-300M? Ngược lại, đối với quá trình tinh chỉnh (fine-tuning) trên các tác vụ phân loại, bộ thông số này thay đổi thành gì?"
    print(f"\n3. Running workflow with query: '{query}'")
    
    # Use the default ViT doc_id
    doc_id = "90034a5d-b97f-4a34-bacc-ef11780f5c89"
    
    start_time = time_time = int(os.getenv("START_TIME", 0)) or None
    import time
    start = time.time()
    final_state = workflow.run(query=query, doc_id=doc_id)
    duration = time.time() - start
    
    print("\n============================================================")
    print("                     TEST EXECUTION RESULT")
    print("============================================================")
    print(f"Total Execution Time: {duration:.2f} seconds")
    print(f"Verification Retries Run: {final_state.get('verification_retries', 0)}")
    print(f"Final Tasks Count: {len(final_state.get('tasks', []))}")
    print("\n--- Tasks Executed ---")
    for idx, task in enumerate(final_state.get("tasks", [])):
        print(f"Task {idx+1} [{task.agent_type.value}]: {task.content}")
        
    print("\n--- Final Answer ---")
    print(final_state.get("final_answer"))
    print("============================================================")

if __name__ == "__main__":
    test_workflow()
