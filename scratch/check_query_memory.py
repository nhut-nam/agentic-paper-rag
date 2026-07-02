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

def check_db():
    db = DatabaseHandler()
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            # Check latest query
            cur.execute("SELECT id, query, response, created_at FROM query_memory ORDER BY created_at DESC LIMIT 1")
            query_row = cur.fetchone()
            if not query_row:
                print("No query memory records found.")
                return
            
            q_id, query, response, q_time = query_row
            print(f"Latest Query ID: {q_id}")
            print(f"Query: {query}")
            print(f"Response (truncated): {response[:200]}...")
            print(f"Created At: {q_time}")
            print("-" * 50)
            
            # Check tasks for this query
            cur.execute("SELECT id, content, task_order, agent_type, status, result, metadata FROM tasks WHERE query_id = %s ORDER BY task_order ASC", (q_id,))
            tasks = cur.fetchall()
            print(f"Found {len(tasks)} tasks:")
            for t in tasks:
                t_id, t_content, t_order, t_agent, t_status, t_result, t_metadata = t
                print(f"  Task #{t_order} (ID: {t_id}) [{t_agent}] status: {t_status}")
                print(f"    Content: {t_content}")
                print(f"    Result: {t_result}")
                print(f"    Metadata: {t_metadata}")
                
                # Check steps for this task
                cur.execute("SELECT step_order, thought, action_name, action_input, observation FROM steps WHERE task_id = %s ORDER BY step_order ASC", (t_id,))
                steps = cur.fetchall()
                print(f"    Found {len(steps)} ReAct steps:")
                for s in steps:
                    s_order, s_thought, s_action, s_input, s_obs = s
                    print(f"      Step {s_order}: Thought: {s_thought[:100]}...")
                    print(f"              Action: {s_action} with input {s_input}")
                    print(f"              Observation (truncated): {s_obs[:100]}...")
                print()
    finally:
        conn.close()

if __name__ == "__main__":
    check_db()
