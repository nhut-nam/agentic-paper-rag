import sys
import os
from dotenv import load_dotenv

# Reconfigure stdout to use UTF-8 on Windows
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.db import DatabaseHandler
from app.pipelines.retrieve import RetrievePipeline
from app.llm.embeddings import embedding_service

def main():
    db = DatabaseHandler()
    doc_id = "90034a5d-b97f-4a34-bacc-ef11780f5c89"
    
    # 1. Count total chunks for this doc
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = %s", (doc_id,))
            count = cur.fetchone()[0]
            print(f"Total chunks in DB for doc_id '{doc_id}': {count}")
            
            # Fetch specifically ch_18
            cur.execute(
                "SELECT content FROM chunks WHERE chunk_id = '90034a5d-b97f-4a34-bacc-ef11780f5c89_ch_18'"
            )
            row = cur.fetchone()
            if row:
                print("\n=== CONTENT OF CHUNK ch_18 ===")
                print(row[0])
                print("==============================\n")
            
            # 2. Search for chunks containing "Table 3" or "Optimizer" or "Batch size"
            cur.execute(
                "SELECT chunk_id, content FROM chunks WHERE doc_id = %s AND (content ILIKE %s OR content ILIKE %s OR content ILIKE %s)",
                (doc_id, '%Table 3%', '%optimizer%', '%batch size%')
            )
            rows = cur.fetchall()
            print(f"Found {len(rows)} chunks containing keywords in DB:")
            for r in rows:
                print(f"- Chunk ID: {r[0]}")
                print(f"  Content snippet: {r[1][:300]}...")
                
    finally:
        conn.close()
        
    # 3. Test retrieve pipeline directly with a clear query
    print("\nRunning RetrievePipeline on query: 'What optimizer, batch size, and learning rate were used during ViT pre-training?'")
    retriever = RetrievePipeline()
    
    # Run hybrid search manually first to check scores
    query = "Adam 4096"
    query_keywords = ['adam', '4096']
    query_vector = retriever.db.hybrid_search(
        embedding_service.encode(query),
        query_keywords,
        limit=15,
        doc_id=doc_id
    )
    
    print("\n--- Raw Hybrid Search Results (Top 15 candidates) ---")
    for idx, row in enumerate(query_vector):
        print(f"Candidate {idx+1}: Chunk ID: {row['chunk_id']} | Vector Score: {row['vector_score']:.4f} | Keyword Score: {row['keyword_score']} | Combined: {(row['vector_score']*0.7 + row['keyword_score']*0.1):.4f}")
    
    results = retriever.run(query, top_k=15, doc_id=doc_id)
    
    print(f"\n--- Reranked Results (All 15 candidates) ---")
    for idx, res in enumerate(results):
        chunk = res["chunk"]
        print(f"Rank {idx+1}: Chunk ID: {chunk.chunk_id} | Combined Score: {res['combined_score']:.4f} | Snippet: {chunk.content[:150]}...")

if __name__ == '__main__':
    main()
