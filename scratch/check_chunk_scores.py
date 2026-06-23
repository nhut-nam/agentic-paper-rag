import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.db import DatabaseHandler
from app.llm.embeddings import embedding_service
from app.pipelines.retrieve import RetrievePipeline

def test_query(query, keywords=None):
    retriever = RetrievePipeline()
    doc_id = "90034a5d-b97f-4a34-bacc-ef11780f5c89"
    
    results = retriever.run(query, top_k=15, doc_id=doc_id)
    
    print(f"\nQUERY: '{query}'")
    print("-" * 60)
    found_rank = -1
    for idx, res in enumerate(results):
        chunk = res["chunk"]
        if "ch_8" in chunk.chunk_id:
            found_rank = idx + 1
            print(f"--> FOUND ch_8 at Reranked Rank {found_rank}! Combined Score: {res['combined_score']:.4f} | Snippet: {chunk.content[:150]}")
            break
    if found_rank == -1:
        print("ch_8 NOT found in top 15 reranked results!")
        # Print all retrieved chunk ids to see what is there
        print("Top 15 retrieved chunk ids:")
        for idx, res in enumerate(results):
            print(f"  {idx+1}: {res['chunk'].chunk_id} (Score: {res['combined_score']:.4f}) - {res['chunk'].heading_path}")

def main():
    # 1. Test original query
    test_query("pre-training optimizer and batch size JFT-300M dataset")
    
    # 2. Test verifier keywords query
    test_query("optimizer batch size pre-training fine-tuning JFT-300M configuration")
    
    # 3. Test just semantic query
    test_query("Adam optimizer batch size 4096 pre-training setup")

if __name__ == "__main__":
    main()
