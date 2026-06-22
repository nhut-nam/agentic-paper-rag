import psycopg2
from psycopg2.extras import RealDictCursor
from app.config.settings import settings
from typing import List, Dict, Any, Optional
from app.utils.logger import logger
from app.models.document import Document
from app.models.chunk import Chunk, ChunkCreate

class DatabaseHandler:
    def __init__(self):
        self.conn_params = {
            "dbname": settings.DB_NAME,
            "user": settings.DB_USER,
            "password": settings.DB_PASSWORD,
            "host": settings.DB_HOST,
            "port": settings.DB_PORT
        }
        self.init_tables()

    def init_tables(self):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # 1. Create session_chat table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS session_chat (
                        id VARCHAR(255) PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # 2. Create session_chat_documents table (Join table)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS session_chat_documents (
                        session_id VARCHAR(255) REFERENCES session_chat(id) ON DELETE CASCADE,
                        doc_id VARCHAR(255) REFERENCES documents(doc_id) ON DELETE CASCADE,
                        PRIMARY KEY (session_id, doc_id)
                    );
                """)
                
                # 3. Create query_memory table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS query_memory (
                        id SERIAL PRIMARY KEY,
                        doc_id VARCHAR(255),
                        query TEXT NOT NULL,
                        response TEXT NOT NULL,
                        retrieved_contexts TEXT[] NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Migration helper: Check if session_id column exists in query_memory, if not, add it
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='query_memory' AND column_name='session_id';
                """)
                if not cur.fetchone():
                    logger.info("Migrating query_memory table: adding session_id column.")
                    cur.execute("""
                        ALTER TABLE query_memory 
                        ADD COLUMN session_id VARCHAR(255) REFERENCES session_chat(id) ON DELETE CASCADE;
                    """)
                    
                # 4. Create summary_session table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS summary_session (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) REFERENCES session_chat(id) ON DELETE CASCADE,
                        summary TEXT NOT NULL,
                        memory_id INT REFERENCES query_memory(id) ON DELETE SET NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # 5. Create document_images table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS document_images (
                        id SERIAL PRIMARY KEY,
                        image_id VARCHAR(255) UNIQUE NOT NULL,
                        doc_id VARCHAR(255) NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                        image_path TEXT NOT NULL,
                        image_type VARCHAR(50) NOT NULL,
                        content TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            conn.commit()
            logger.info("Initialized system tables (session_chat, session_chat_documents, query_memory, summary_session, document_images) successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize system tables: {e}")
            conn.rollback()
        finally:
            conn.close()

    def get_connection(self):
        try:
            return psycopg2.connect(**self.conn_params)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def insert_document(self, doc_id: str, path: str):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO documents (doc_id, path) VALUES (%s, %s) ON CONFLICT (doc_id) DO UPDATE SET path = EXCLUDED.path, status = 'uploaded'",
                    (doc_id, path)
                )
            conn.commit()
        finally:
            conn.close()

    def update_status(self, doc_id: str, status: str):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET status = %s, updated_at = CURRENT_TIMESTAMP WHERE doc_id = %s",
                    (status, doc_id)
                )
            conn.commit()
        finally:
            conn.close()

    def get_document(self, doc_id: str) -> Optional[Document]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM documents WHERE doc_id = %s", (doc_id,))
                row = cur.fetchone()
                return Document(**row) if row else None
        finally:
            conn.close()

    def get_all_documents(self) -> List[Document]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM documents ORDER BY created_at DESC")
                rows = cur.fetchall()
                return [Document(**row) for row in rows]
        finally:
            conn.close()

    def delete_document(self, doc_id: str):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
                cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
            conn.commit()
            logger.info(f"Deleted document and its chunks for doc_id: {doc_id}")
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_chunks_by_doc_id(self, doc_id: str) -> List[Chunk]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM chunks WHERE doc_id = %s ORDER BY chunk_order ASC", (doc_id,))
                rows = cur.fetchall()
                
                # Convert embedding string to list for Pydantic validation
                processed_rows = []
                for row in rows:
                    if row.get("embedding") and isinstance(row["embedding"], str):
                        # pgvector returns '[0.1, 0.2, ...]'
                        vec_str = row["embedding"].strip("[]")
                        row["embedding"] = [float(x) for x in vec_str.split(",")]
                    processed_rows.append(row)
                    
                return [Chunk(**row) for row in processed_rows]
        finally:
            conn.close()

    def hybrid_search(self, query_vector: list, query_keywords: list, limit: int = 5, doc_id: Optional[str] = None, doc_ids: Optional[List[str]] = None) -> List[dict]:
        # Merge doc_id and doc_ids into a unique list
        target_ids = []
        if doc_ids:
            target_ids.extend(doc_ids)
        if doc_id and doc_id not in target_ids:
            target_ids.append(doc_id)

        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Weighted score: 70% Vector similarity + 30% Keyword overlap
                if target_ids:
                    sql = """
                    SELECT *, 
                           (1 - (embedding <=> %s::vector)) AS vector_score,
                           (SELECT count(*) FROM unnest(keywords) k WHERE k = ANY(%s)) AS keyword_score
                    FROM chunks
                    WHERE doc_id = ANY(%s)
                    ORDER BY ((1 - (embedding <=> %s::vector)) * 0.7 + 
                             (SELECT count(*) FROM unnest(keywords) k WHERE k = ANY(%s)) * 0.3) DESC
                    LIMIT %s
                    """
                    cur.execute(sql, (query_vector, query_keywords, target_ids, query_vector, query_keywords, limit))
                else:
                    sql = """
                    SELECT *, 
                           (1 - (embedding <=> %s::vector)) AS vector_score,
                           (SELECT count(*) FROM unnest(keywords) k WHERE k = ANY(%s)) AS keyword_score
                    FROM chunks
                    ORDER BY ((1 - (embedding <=> %s::vector)) * 0.7 + 
                             (SELECT count(*) FROM unnest(keywords) k WHERE k = ANY(%s)) * 0.3) DESC
                    LIMIT %s
                    """
                    cur.execute(sql, (query_vector, query_keywords, query_vector, query_keywords, limit))
                return cur.fetchall()
        finally:
            conn.close()
    def insert_chunk(self, chunk_data: dict):
        """
        Inserts a chunk into the database. 
        Accepts a dict for flexibility during the pipeline, 
        but validates it internally.
        """
        # Validate data using Pydantic
        chunk = ChunkCreate(**chunk_data)
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, doc_id, content, heading_path, section_title, 
                        chunk_order, summary, keywords, embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET 
                        content = EXCLUDED.content,
                        heading_path = EXCLUDED.heading_path,
                        section_title = EXCLUDED.section_title,
                        summary = EXCLUDED.summary,
                        keywords = EXCLUDED.keywords,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        chunk.chunk_id, 
                        chunk.doc_id, 
                        chunk.content, 
                        chunk.heading_path, 
                        chunk.section_title, 
                        chunk.chunk_order,
                        chunk.summary,
                        chunk.keywords,
                        chunk.embedding
                    )
                )
            conn.commit()
        finally:
            conn.close()

    def insert_query_memory(self, query: str, response: str, retrieved_contexts: List[str], doc_id: Optional[str] = None, session_id: Optional[str] = None) -> Optional[int]:
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO query_memory (query, response, retrieved_contexts, doc_id, session_id) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (query, response, retrieved_contexts, doc_id, session_id)
                )
                memory_id = cur.fetchone()[0]
            conn.commit()
            logger.info(f"Inserted interaction into query_memory with ID: {memory_id}.")
            return memory_id
        except Exception as e:
            logger.error(f"Failed to insert query memory: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_all_query_memory(self, limit: int = 100, doc_id: Optional[str] = None, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if session_id:
                    cur.execute(
                        "SELECT * FROM query_memory WHERE session_id = %s ORDER BY created_at DESC LIMIT %s",
                        (session_id, limit)
                    )
                elif doc_id:
                    cur.execute(
                        "SELECT * FROM query_memory WHERE doc_id = %s ORDER BY created_at DESC LIMIT %s",
                        (doc_id, limit)
                    )
                else:
                    cur.execute("SELECT * FROM query_memory ORDER BY created_at DESC LIMIT %s", (limit,))
                return cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to get query memory: {e}")
            raise
        finally:
            conn.close()

    def get_query_memory_by_id(self, memory_id: int) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM query_memory WHERE id = %s", (memory_id,))
                return cur.fetchone()
        except Exception as e:
            logger.error(f"Failed to get query memory by id {memory_id}: {e}")
            raise
        finally:
            conn.close()

    def create_session_chat(self, session_id: str):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO session_chat (id) VALUES (%s) ON CONFLICT (id) DO NOTHING",
                    (session_id,)
                )
            conn.commit()
            logger.info(f"Session chat {session_id} created or verified.")
        except Exception as e:
            logger.error(f"Failed to create session_chat: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def link_session_document(self, session_id: str, doc_id: str):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO session_chat_documents (session_id, doc_id) VALUES (%s, %s) ON CONFLICT (session_id, doc_id) DO NOTHING",
                    (session_id, doc_id)
                )
            conn.commit()
            logger.info(f"Linked session chat {session_id} with document {doc_id}.")
        except Exception as e:
            logger.error(f"Failed to link session_document: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_session_documents(self, session_id: str) -> List[dict]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT d.* 
                    FROM documents d
                    JOIN session_chat_documents sd ON d.doc_id = sd.doc_id
                    WHERE sd.session_id = %s
                    ORDER BY d.created_at DESC
                    """,
                    (session_id,)
                )
                return cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to get session documents for session {session_id}: {e}")
            return []
        finally:
            conn.close()

    def get_latest_summary_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM summary_session WHERE session_id = %s ORDER BY created_at DESC LIMIT 1",
                    (session_id,)
                )
                return cur.fetchone()
        except Exception as e:
            logger.error(f"Failed to get latest summary_session for session {session_id}: {e}")
            raise
        finally:
            conn.close()

    def insert_summary_session(self, session_id: str, summary: str, memory_id: Optional[int] = None):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO summary_session (session_id, summary, memory_id) VALUES (%s, %s, %s)",
                    (session_id, summary, memory_id)
                )
            conn.commit()
            logger.info(f"Inserted summary into summary_session for session: {session_id}.")
        except Exception as e:
            logger.error(f"Failed to insert summary_session: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def insert_document_image(self, image_id: str, doc_id: str, image_path: str, image_type: str, content: Optional[str] = None):
        """
        Inserts or updates an image metadata in document_images table.
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO document_images (image_id, doc_id, image_path, image_type, content)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (image_id) DO UPDATE SET
                        image_path = EXCLUDED.image_path,
                        image_type = EXCLUDED.image_type,
                        content = EXCLUDED.content
                    """,
                    (image_id, doc_id, image_path, image_type, content)
                )
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to insert document image metadata for {image_id}: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_images_by_ids(self, image_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetches metadata for a list of image_ids.
        """
        if not image_ids:
            return []
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT image_id, doc_id, image_path, image_type, content FROM document_images WHERE image_id = ANY(%s)",
                    (image_ids,)
                )
                return cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to fetch images by ids {image_ids}: {e}")
            return []
        finally:
            conn.close()
