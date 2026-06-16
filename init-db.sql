CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(255) UNIQUE NOT NULL,
    path TEXT NOT NULL,
    status VARCHAR(50) DEFAULT 'uploaded',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS document_images (
    id SERIAL PRIMARY KEY,
    image_id VARCHAR(255) UNIQUE NOT NULL,
    doc_id VARCHAR(255) NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    image_path TEXT NOT NULL,
    image_type VARCHAR(50) NOT NULL,
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(255) UNIQUE NOT NULL,
    doc_id VARCHAR(255) NOT NULL REFERENCES documents(doc_id),
    
    content TEXT NOT NULL,         
    summary TEXT,                 
    keywords TEXT[],               
    
    heading_path TEXT,             
    section_title TEXT,
    page_ref TEXT,                 
    chunk_order INT,
    
    embedding VECTOR(384),        
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE plans (
    id VARCHAR(255) PRIMARY KEY,
    query TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE tasks (
    id VARCHAR(255) PRIMARY KEY,
    plan_id VARCHAR(255) REFERENCES plans(id),
    content TEXT NOT NULL,
    task_order INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE steps (
    id VARCHAR(255) PRIMARY KEY,
    task_id VARCHAR(255) REFERENCES tasks(id),
    agent_type VARCHAR(50),
    status VARCHAR(50) DEFAULT 'pending',
    result JSONB,
    context JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS session_chat (
    id VARCHAR(255) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS session_chat_documents (
    session_id VARCHAR(255) REFERENCES session_chat(id) ON DELETE CASCADE,
    doc_id VARCHAR(255) REFERENCES documents(doc_id) ON DELETE CASCADE,
    PRIMARY KEY (session_id, doc_id)
);

CREATE TABLE IF NOT EXISTS query_memory (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) REFERENCES session_chat(id) ON DELETE CASCADE,
    doc_id VARCHAR(255) REFERENCES documents(doc_id) ON DELETE SET NULL,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    retrieved_contexts TEXT[] NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS summary_session (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) REFERENCES session_chat(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    memory_id INT REFERENCES query_memory(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);