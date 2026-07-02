import os
import sys
from dotenv import load_dotenv

# Reconfigure stdout to use UTF-8 on Windows
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.db import DatabaseHandler

def dump_db():
    db = DatabaseHandler()
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, query_id, task_order, agent_type, status FROM tasks")
            tasks = cur.fetchall()
            print(f"Total tasks in DB: {len(tasks)}")
            for t in tasks:
                print(f"  Task: {t}")
                
            cur.execute("SELECT id, task_id, step_order, action_name FROM steps")
            steps = cur.fetchall()
            print(f"Total steps in DB: {len(steps)}")
            for s in steps:
                print(f"  Step: {s}")
    finally:
        conn.close()

if __name__ == "__main__":
    dump_db()
