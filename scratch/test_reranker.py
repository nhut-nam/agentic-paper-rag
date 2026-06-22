import sys
import io
import time
import uuid
from app.utils.db import DatabaseHandler
from app.pipelines.retrieve import RetrievePipeline
from app.llm.embeddings import embedding_service
from app.utils.logger import logger

# Reconfigure stdout/stderr to use UTF-8 on Windows to avoid UnicodeEncodeErrors
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def main():
    print("=== BAT DAU KIEM THU RERANKER & BYPASS HEURISTIC ===")
    
    db = DatabaseHandler()
    
    doc_id = "test-doc-reranker"
    # Dọn dẹp dữ liệu cũ (nếu có)
    try:
        db.delete_document(doc_id)
        print("Da don dep du lieu cu cua doc_id test.")
    except Exception:
        pass
        
    # 1. Chen document thu nghiem
    print(f"Dang chen document test: {doc_id}")
    db.insert_document(doc_id, path="/path/to/test-doc-reranker.pdf")
    
    # 2. Chuan bi 3 chunks mau
    chunk_data_list = [
        {
            "chunk_id": f"chunk-1-{uuid.uuid4().hex[:6]}",
            "doc_id": doc_id,
            "content": "Vision Transformer (ViT) represents a milestone in computer vision. It applies Transformer architectures directly to image patches for image classification tasks.",
            "heading_path": "Introduction / Vision Transformer",
            "section_title": "Vision Transformer Overview",
            "chunk_order": 1,
            "summary": "Introduction to ViT and patches.",
            # Tối ưu hóa keywords để khớp chính xác với các n-gram mà KeyBERT trích xuất từ Case 1 query
            "keywords": ["vision transformer", "patches classification", "transformer vit", "vit patches", "transformer", "vision", "vit", "patches", "classification"]
        },
        {
            "chunk_id": f"chunk-2-{uuid.uuid4().hex[:6]}",
            "doc_id": doc_id,
            "content": "Reranking is a common technique in information retrieval. Cross-encoder models score query and document pairs together to capture deep semantic matches.",
            "heading_path": "Retrieval / Reranker",
            "section_title": "Cross-Encoder Reranker",
            "chunk_order": 2,
            "summary": "How reranking with cross-encoder works.",
            "keywords": ["reranking", "information", "retrieval", "cross-encoder", "semantic"]
        },
        {
            "chunk_id": f"chunk-3-{uuid.uuid4().hex[:6]}",
            "doc_id": doc_id,
            "content": "Keyword search operates by matching exact terms in documents. However, it often misses semantic synonyms that vector search can easily find.",
            "heading_path": "Retrieval / Hybrid",
            "section_title": "Keyword vs Vector Search",
            "chunk_order": 3,
            "summary": "Keyword matching limitations compared to vector search.",
            "keywords": ["keyword", "search", "exact", "matching", "synonyms"]
        }
    ]
    
    print("Dang sinh embedding va chen 3 chunks vao DB...")
    for data in chunk_data_list:
        data["embedding"] = embedding_service.encode(data["content"])
        db.insert_chunk(data)
        print(f"Da chen: {data['chunk_id']} | Keywords: {data['keywords']}")
        
    print("Khoi tao RetrievePipeline...")
    pipeline = RetrievePipeline()
    
    # CASE 1: Query co keyword overlap cao (Kỳ vọng: Bypass Rerank)
    query1 = "Tell me about Vision Transformer ViT patches and classification."
    print(f"\n--- CASE 1: Query co keyword score cao (Ky vong: BYPASS RERANK) ---")
    print(f"Query: '{query1}'")
    
    start_time = time.time()
    results1 = pipeline.run(query=query1, top_k=3, doc_id=doc_id)
    latency1 = time.time() - start_time
    
    print(f"Thoi gian phan hoi (Latency): {latency1:.4f} giay")
    print(f"So luong ket qua tra ve: {len(results1)}")
    for i, res in enumerate(results1):
        chunk = res["chunk"]
        print(f" Rank {i+1}: ID={chunk.chunk_id}")
        print(f"  Content: {chunk.content[:80]}...")
        print(f"  Vector Score: {res.get('vector_score'):.4f} | Keyword Score: {res.get('keyword_score')} | Combined Score: {res.get('combined_score'):.4f}")
        print(f"  Rerank Score (Neu co): {res.get('rerank_score')}")
        
    # CASE 2: Query co keyword overlap thap (Kỳ vọng: Run Reranker)
    query2 = "How does deep semantic matching work in retrieval system?"
    print(f"\n--- CASE 2: Query co keyword score thap (Ky vong: RUN RERANKER) ---")
    print(f"Query: '{query2}'")
    
    start_time = time.time()
    results2 = pipeline.run(query=query2, top_k=3, doc_id=doc_id)
    latency2 = time.time() - start_time
    
    print(f"Thoi gian phan hoi (Latency): {latency2:.4f} giay")
    print(f"So luong ket qua tra ve: {len(results2)}")
    for i, res in enumerate(results2):
        chunk = res["chunk"]
        print(f" Rank {i+1}: ID={chunk.chunk_id}")
        print(f"  Content: {chunk.content[:80]}...")
        print(f"  Vector Score: {res.get('vector_score'):.4f} | Keyword Score: {res.get('keyword_score')} | Combined Score: {res.get('combined_score'):.4f}")
        print(f"  Rerank Score (Neu co): {res.get('rerank_score')}")
        
    # 3. Don dep
    print(f"\nDang xoa document test va cac chunk lien quan...")
    db.delete_document(doc_id)
    print("Don dep hoan tat!")
    print("=== KET THUC KIEM THU ===")

if __name__ == "__main__":
    main()
