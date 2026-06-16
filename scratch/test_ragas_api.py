import sys
import io
import requests
import json
import time

# Reconfigure stdout/stderr to use UTF-8 on Windows to avoid UnicodeEncodeErrors
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_URL = "http://127.0.0.1:8000"

def wait_for_server():
    print("Waiting for API server to become ready...")
    for i in range(15):
        try:
            response = requests.get(f"{BASE_URL}/documents")
            if response.status_code == 200:
                print("API Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    print("API Server did not start in time.")
    return False

def test_flow():
    if not wait_for_server():
        return
        
    # 1. List documents to find a valid doc_id
    print("\n--- 1. Fetching Documents ---")
    doc_res = requests.get(f"{BASE_URL}/documents")
    docs = doc_res.json()
    print(f"Documents found: {json.dumps(docs, indent=2, ensure_ascii=False)}")
    
    doc_id = None
    if docs:
        doc_id = docs[0].get("doc_id")
        print(f"Using existing doc_id: {doc_id}")
    else:
        print("No documents found in DB. Agent query will run without document scope.")

    # 2. Trigger an agent query (which will be logged to query_memory)
    print("\n--- 2. Sending Agent Queries (to populate query_memory) ---")
    queries = [
        "What is the core contribution of the DaViT model?",
        "Mô tả cấu hình và các tham số chính của DaViT-Small là gì?"
    ]
    
    last_memory_id = None
    for q in queries:
        print(f"\nSending Query: '{q}'...")
        payload = {"query": q}
        if doc_id:
            payload["doc_id"] = doc_id
            
        try:
            res = requests.post(f"{BASE_URL}/agent/query", params=payload)
            if res.status_code == 200:
                res_data = res.json()
                last_memory_id = res_data.get("memory_id")
                print(f"Memory ID: {last_memory_id}")
                print(f"Response: {res_data.get('final_answer')[:200]}...")
            else:
                print(f"Error {res.status_code}: {res.text}")
        except Exception as e:
            print(f"Request failed: {e}")

    # 3. Call evaluate-memory endpoint
    print("\n--- 3. Running Ragas Evaluation on Memory ---")
    eval_payload = {}
    if last_memory_id:
        eval_payload["memory_id"] = last_memory_id
        print(f"Calling /evaluate-memory for Memory ID {last_memory_id}...")
    else:
        eval_payload["limit"] = 5
        if doc_id:
            eval_payload["doc_id"] = doc_id
        print("Calling /evaluate-memory...")
    try:
        res = requests.post(f"{BASE_URL}/evaluate-memory", params=eval_payload)
        if res.status_code == 200:
            results = res.json()
            print("\n=== Evaluation Results ===")
            print(f"Message: {results.get('message')}")
            print(f"Evaluated Count: {results.get('evaluated_count')}")
            print(f"Overall Scores: {json.dumps(results.get('overall_scores'), indent=2)}")
            print("\nDetails per Row:")
            for detail in results.get("details", []):
                question = detail.get('question') or "None"
                answer = detail.get('answer') or "None"
                print(f"- Q: {question}")
                print(f"  A: {answer[:100]}...")
                print(f"  Scores: {json.dumps(detail.get('scores'))}")
        else:
            print(f"Error {res.status_code}: {res.text}")
    except Exception as e:
        print(f"Evaluation request failed: {e}")

if __name__ == "__main__":
    test_flow()
