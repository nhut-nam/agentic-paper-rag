import os
import sys
import shutil
import psycopg2

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.settings import settings

def main():
    # 1. Clean up file system
    storage_dirs = [
        os.path.join("storage", "uploads"),
        os.path.join("storage", "processed")
    ]

    for directory in storage_dirs:
        if os.path.exists(directory):
            print(f"Clearing directory: {directory}")
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        else:
            os.makedirs(directory, exist_ok=True)

    # 2. Clear database tables
    conn_params = {
        "dbname": settings.DB_NAME,
        "user": settings.DB_USER,
        "password": settings.DB_PASSWORD,
        "host": settings.DB_HOST,
        "port": settings.DB_PORT
    }

    try:
        conn = psycopg2.connect(**conn_params)
        with conn.cursor() as cur:
            # Use CASCADE to handle foreign key dependencies
            cur.execute("TRUNCATE TABLE chunks, documents, query_memory CASCADE;")
            print("Successfully truncated all database tables.")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database truncate failed: {e}")

if __name__ == "__main__":
    main()
