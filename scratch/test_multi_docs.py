import sys
import io
import time
import uuid
import requests
from app.utils.db import DatabaseHandler
from app.pipelines.retrieve import RetrievePipeline
from app.llm.embeddings import embedding_service
from app.utils.logger import logger

# Reconfigure stdout/stderr to use UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def main():
    print("=== BAT DAU KIEM THU RETRIEVAL & AGENT DA TAI LIEU (MULTI-DOCUMENTS) ===")
    
    db = DatabaseHandler()
    
    doc_id_1 = "test-multi-doc-1"
    doc_id_2 = "test-multi-doc-2"
    
    # 1. Don dep du lieu cu (neu co)
    try:
        db.delete_document(doc_id_1)
        db.delete_document(doc_id_2)
        print("Da don dep du lieu cu.")
    except Exception:
        pass
        
    # 2. Chen documents test
    print("Dang chen documents...")
    db.insert_document(doc_id_1, path="/path/to/attention_is_all_you_need.pdf")
    db.insert_document(doc_id_2, path="/path/to/vit_paper.pdf")
    
    # 3. Chen chunks
    chunk_1 = {
        "chunk_id": f"chunk-attn-{uuid.uuid4().hex[:6]}",
        "doc_id": doc_id_1,
        "content": "Attention mechanism allows the model to focus on specific parts of the input sequence. The Transformer relies entirely on Self-Attention to compute representations without recurrent layers.",
        "heading_path": "Architecture / Attention",
        "section_title": "Self-Attention Mechanism",
        "chunk_order": 1,
        "summary": "Details about self-attention in Transformer.",
        "keywords": ["attention", "self-attention", "mechanism", "transformer"]
    }
    
    chunk_2 = {
        "chunk_id": f"chunk-pos-{uuid.uuid4().hex[:6]}",
        "doc_id": doc_id_2,
        "content": "Vision Transformer (ViT) applies a standard Transformer encoder directly to patches of images. An image is split into fixed-size patches and projected linearly before entering the encoder.",
        "heading_path": "Vision Transformer / Patches",
        "section_title": "Image Patch Projection",
        "chunk_order": 1,
        "summary": "How image patches are processed in ViT.",
        "keywords": ["vision", "transformer", "vit", "patches", "projection"]
    }
    
    print("Dang sinh embedding va chen chunks...")
    for chunk_data in [chunk_1, chunk_2]:
        chunk_data["embedding"] = embedding_service.encode(chunk_data["content"])
        db.insert_chunk(chunk_data)
        print(f"Da chen: {chunk_data['chunk_id']} cho Doc: {chunk_data['doc_id']}")
        
    # 4. Test Retrieve Pipeline tren ca 2 docs
    print("\n--- Buoc 1: Test Retrieve Pipeline tren ca hai tai lieu (doc_ids) ---")
    pipeline = RetrievePipeline()
    query = "Explain attention mechanism and vision transformer patches."
    print(f"Query: '{query}'")
    
    results = pipeline.run(query=query, top_k=5, doc_ids=[doc_id_1, doc_id_2])
    print(f"So luong ket qua tim thay: {len(results)}")
    
    found_docs = set()
    for i, res in enumerate(results):
        c = res["chunk"]
        found_docs.add(c.doc_id)
        print(f" Rank {i+1}: Doc ID={c.doc_id} | Chunk ID={c.chunk_id}")
        print(f"  Content: {c.content[:100]}...")
        print(f"  Combined Score: {res.get('combined_score'):.4f}")
        
    if doc_id_1 in found_docs and doc_id_2 in found_docs:
        print("SUCCESS: Tim thay dong thoi cac chunk tu ca hai tai lieu!")
    else:
        print("FAILURE: Khong tim thay du ca hai tai lieu.")
        
    # 5. Test Agent API /agent/query qua request
    # Truoc het, can dam bao API server backend dang chay!
    # Script nay se goi API cục bộ
    print("\n--- Buoc 2: Goi API /agent/query cuc bo ---")
    url = "http://127.0.0.1:8000/agent/query"
    params = {
        "query": "Compare Self-Attention mechanism in Transformer and how patches are used in Vision Transformer.",
        "doc_ids": f"{doc_id_1},{doc_id_2}",
        "session_id": f"test-session-multi-{uuid.uuid4().hex[:6]}"
    }
    
    try:
        print(f"Dang gui yeu cau toi {url} vao doc_ids='{params['doc_ids']}'...")
        res = requests.post(url, params=params, timeout=30)
        if res.status_code == 200:
            res_data = res.json()
            print("API Response OK!")
            print(f"Mode: {res_data.get('mode')}")
            print(f"Final Answer:\n{res_data.get('final_answer')}")
            print(f"Retrieved Context Count: {len(res_data.get('retrieved_docs', []))}")
            if len(res_data.get("retrieved_docs", [])) >= 2:
                print("SUCCESS: Agent da su dung context tu ca hai tai lieu de phan tich!")
            else:
                print("WARNING: So luong context thu hoi thap.")
        else:
            print(f"API Server loi {res.status_code}: {res.text}")
            print("Luu y: Đảm bảo bạn đã khởi chạy FastAPI server (uvicorn) truoc khi chay test nay.")
    except Exception as e:
        print(f"Khong the ket noi den API Server: {e}")
        print("Luu y: Vui long chay API Server o terminal khac de buoc test nay hoan thanh!")
        
    # 6. Don dep
    print("\nDang don dep du lieu test...")
    db.delete_document(doc_id_1)
    db.delete_document(doc_id_2)
    print("Don dep hoan tat.")
    print("=== KET THUC KIEM THU ===")

if __name__ == "__main__":
    main()
