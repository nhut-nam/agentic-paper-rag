import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.db import DatabaseHandler

def main():
    db = DatabaseHandler()
    doc_id = "90034a5d-b97f-4a34-bacc-ef11780f5c89"
    
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, heading_path, content FROM chunks WHERE doc_id = %s ORDER BY chunk_id",
                (doc_id,)
            )
            rows = cur.fetchall()
            
            output_path = "scratch/all_vit_chunks.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Total chunks: {len(rows)}\n\n")
                for r in rows:
                    f.write("============================================================\n")
                    f.write(f"CHUNK ID: {r[0]} | SECTION: {r[1]}\n")
                    f.write("============================================================\n")
                    f.write(r[2])
                    f.write("\n\n")
            print(f"Successfully saved all {len(rows)} chunks to {output_path}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
