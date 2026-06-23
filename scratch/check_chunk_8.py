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

def main():
    db = DatabaseHandler()
    doc_id = "90034a5d-b97f-4a34-bacc-ef11780f5c89"
    chunk_id = "90034a5d-b97f-4a34-bacc-ef11780f5c89_ch_8"
    
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT content, keywords, heading_path, section_title FROM chunks WHERE chunk_id = %s", (chunk_id,))
            row = cur.fetchone()
            if row:
                print(f"=== Chunk ID: {chunk_id} ===")
                print(f"Heading Path : {row[2]}")
                print(f"Section Title: {row[3]}")
                print(f"Keywords     : {row[1]}")
                print(f"--- CONTENT ---")
                print(row[0])
            else:
                print("Chunk not found.")
    finally:
        conn.close()

if __name__ == '__main__':
    main()
