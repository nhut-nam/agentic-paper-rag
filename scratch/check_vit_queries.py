import sys
from app.utils.db import DatabaseHandler

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
        
    db = DatabaseHandler()
    VIT_DOC_ID = "90034a5d-b97f-4a34-bacc-ef11780f5c89"
    
    # Get 50 query memories for ViT
    memory = db.get_all_query_memory(limit=50, doc_id=VIT_DOC_ID)
    if not memory:
        print("Không tìm thấy dữ liệu query memory nào cho ViT.")
        return
        
    print(f"Tìm thấy {len(memory)} bản ghi query memory cho ViT.")
    print("=" * 60)
    
    # Reverse memory to display in chronological order (oldest first)
    memory.reverse()
    
    for idx, m in enumerate(memory):
        print(f"#{idx+1} | ID: {m.get('id')} | Created: {m.get('created_at')}")
        print(f"Query: {m.get('query')}")
        print(f"Response: {m.get('response')}")
        print("-" * 60)

if __name__ == '__main__':
    main()
