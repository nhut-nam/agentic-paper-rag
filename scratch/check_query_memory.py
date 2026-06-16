import sys
from app.utils.db import DatabaseHandler

def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass  # In case reconfigure isn't available
    db = DatabaseHandler()
    memory = db.get_all_query_memory(limit=5)
    if not memory:
        print("No query memory entries found in the table.")
        return
        
    for m in memory:
        print(f"ID: {m.get('id')}")
        print(f"Doc ID: {m.get('doc_id')}")
        print(f"Query: {m.get('query')}")
        response = m.get('response', '')
        # Truncate response for display
        resp_trunc = response[:200] + ('...' if len(response) > 200 else '')
        print(f"Response: {resp_trunc}")
        contexts = m.get('retrieved_contexts', [])
        print(f"Retrieved Contexts Count: {len(contexts)}")
        print(f"Created At: {m.get('created_at')}")
        print("-" * 50)

if __name__ == '__main__':
    main()
