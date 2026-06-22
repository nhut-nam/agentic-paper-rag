import os
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from app.pipelines.ingest import IngestPipeline
from app.pipelines.chunking import ChunkingPipeline
from app.pipelines.retrieve import RetrievePipeline
from app.utils.storage import StorageManager
from app.utils.db import DatabaseHandler
from app.models.document import Document
from app.models.chunk import Chunk
from app.models.pipeline_result import IngestResult, ChunkingResult
from app.utils.logger import logger
from typing import List
from app.agent.agent_loop import AgentWorkflow
from app.llm.factory import LLMFactory

app = FastAPI(title="Paper Intelligent AI API")
storage = StorageManager()
db = DatabaseHandler()

@app.get("/documents", response_model=List[Document])
async def list_documents():
    """
    List all documents in the database.
    """
    return db.get_all_documents()

@app.get("/documents/{doc_id}", response_model=Document)
async def get_document_details(doc_id: str):
    """
    Get detailed info for a specific document.
    """
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc

@app.get("/documents/{doc_id}/chunks", response_model=List[Chunk])
async def get_document_chunks(doc_id: str):
    """
    Retrieve all smart chunks for a specific document.
    """
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    return db.get_chunks_by_doc_id(doc_id)

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and register it in the database.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    # Save file using original filename
    relative_path = f"uploads/{file.filename}"
    content = await file.read()
    full_path = storage.save_file(relative_path, content)
    
    # Check if a document with this path is already registered
    existing_docs = db.get_all_documents()
    for d in existing_docs:
        if os.path.abspath(d.path) == os.path.abspath(full_path):
            logger.info(f"Document already registered: {d.doc_id} for path {full_path}")
            return {
                "status": "uploaded",
                "doc_id": d.doc_id,
                "filename": file.filename,
                "storage_path": relative_path
            }
            
    # Generate unique doc_id for new documents
    doc_id = str(uuid.uuid4())
    
    # Save to DB
    try:
        db.insert_document(doc_id, full_path)
    except Exception as e:
        logger.error(f"Failed to register document in DB: {e}")
        raise HTTPException(status_code=500, detail="Database registration failed.")
    
    return {
        "status": "uploaded",
        "doc_id": doc_id,
        "filename": file.filename,
        "storage_path": relative_path
    }

@app.post("/ingest/{doc_id}")
async def trigger_ingest(doc_id: str, background_tasks: BackgroundTasks):
    """
    Trigger the ingestion pipeline for a previously uploaded document.
    Runs in the background.
    """
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    # Check if already processing to avoid duplicates
    if doc.status in ["ingesting", "chunking"]:
        return {"status": "already_processing", "doc_id": doc_id, "current_status": doc.status}

    def run_pipeline(d_id: str, d_path: str):
        # The Master IngestPipeline now handles the full sequence:
        # PDF -> Markdown -> Smart Chunking -> LLM Enrichment
        pipeline = IngestPipeline()
        pipeline.run(d_path, doc_id=d_id)

    background_tasks.add_task(run_pipeline, doc_id, doc.path)
    
    return {
        "status": "ingestion_started",
        "doc_id": doc_id
    }

@app.post("/chunk/{doc_id}")
async def trigger_chunking(doc_id: str, background_tasks: BackgroundTasks):
    """
    Trigger the smart chunking pipeline for an ingested document.
    """
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    # Check if already processing
    if doc.status in ["ingesting", "chunking"]:
        return {"status": "already_processing", "doc_id": doc_id, "current_status": doc.status}

    # Check if markdown file exists on disk to allow chunking
    markdown_path = storage.get_full_path(f"processed/{doc_id}/markdown/result.md")
    if not os.path.exists(markdown_path):
        raise HTTPException(status_code=400, detail="Document markdown not found. Please run ingestion first.")

    def run_chunking(d_id: str):
        pipeline = ChunkingPipeline()
        pipeline.run(d_id)

    background_tasks.add_task(run_chunking, doc_id)
    
    return {
        "status": "chunking_started",
        "doc_id": doc_id
    }

@app.get("/status/{doc_id}", response_model=Document)
async def get_status(doc_id: str):
    """
    Check the status of a document.
    """
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    return doc

@app.post("/retrieve")
async def retrieve_info(query: str, top_k: int = 5):
    """
    Perform hybrid search to retrieve relevant information.
    """
    pipeline = RetrievePipeline()
    results = pipeline.run(query, top_k=top_k)
    return results

@app.post("/agent/query")
async def run_agent_query(query: str, doc_id: str = None, doc_ids: str = None, session_id: str = None):
    """
    Process a query through the Agent Workflow.
    Can be optionally scoped to a specific doc_id, doc_ids (comma-separated), and session_id.
    """
    logger.info(f"API received query for Agent Workflow: {query} (doc_id: {doc_id}, doc_ids: {doc_ids}, session_id: {session_id})")
    try:
        parsed_doc_ids = []
        if doc_ids:
            parsed_doc_ids = [d.strip() for d in doc_ids.split(",") if d.strip()]
        if doc_id and doc_id not in parsed_doc_ids:
            parsed_doc_ids.append(doc_id)

        llm_provider = LLMFactory.get_provider("ollama")
        workflow = AgentWorkflow(llm_provider=llm_provider)
        
        final_state = workflow.run(query, doc_id=doc_id, doc_ids=parsed_doc_ids, session_id=session_id)
        
        # Format output for API response
        response_data = {
            "query": final_state.get("query"),
            "mode": final_state.get("mode"),
            "language": final_state.get("language"),
            "session_id": final_state.get("session_id"),
            "session_summary": final_state.get("session_summary")
        }
        
        plan = final_state.get("plan")
        if plan:
            response_data["plan_id"] = plan.id
            
        response_data["tasks"] = [t.model_dump() for t in final_state.get("tasks", [])]
        response_data["final_answer"] = final_state.get("final_answer")
        response_data["memory_id"] = final_state.get("memory_id")
        response_data["retrieved_docs"] = final_state.get("retrieved_docs", [])
            
        return response_data
        
    except Exception as e:
        logger.error(f"Agent Workflow API failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

eval_llm = None
eval_embeddings = None

def get_evaluation_resources():
    global eval_llm, eval_embeddings
    if eval_llm is None or eval_embeddings is None:
        logger.info("Initializing Ragas evaluation resources (ChatGroq & local Embeddings)...")
        # Load env vars just in case
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="GROQ_API_KEY environment variable is missing. Cannot perform evaluation."
            )
            
        from langchain_groq import ChatGroq
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        eval_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0.0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        eval_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Ragas evaluation resources initialized successfully.")
    return eval_llm, eval_embeddings

@app.post("/evaluate-memory")
async def evaluate_memory(limit: int = 10, doc_id: str = None, memory_id: int = None):
    """
    Run Ragas evaluation over logged query history from database.
    Filter by doc_id or memory_id if provided.
    """
    logger.info(f"API received request to evaluate memory: limit={limit}, doc_id={doc_id}, memory_id={memory_id}")
    import math
    
    # 1. Fetch memory from database
    try:
        if memory_id is not None:
            record = db.get_query_memory_by_id(memory_id)
            records = [record] if record else []
        else:
            records = db.get_all_query_memory(limit=limit, doc_id=doc_id)
    except Exception as e:
        logger.error(f"Failed to fetch query memory for evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Database fetch failed: {str(e)}")
        
    if not records:
        return {
            "message": "No query history found in memory matching the criteria.",
            "evaluated_count": 0,
            "overall_scores": {},
            "details": []
        }
        
    # 2. Format into Ragas Dataset
    questions = []
    contexts_list = []
    answers = []
    
    for row in records:
        questions.append(row["query"])
        ctx = row.get("retrieved_contexts", [])
        if isinstance(ctx, str):
            ctx = [ctx]
        elif not ctx:
            ctx = ["No context retrieved."]
        contexts_list.append(ctx)
        answers.append(row["response"])
        
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        
        test_data = {
            "question": questions,
            "contexts": contexts_list,
            "answer": answers
        }
        dataset = Dataset.from_dict(test_data)
        
        # 3. Get LLM and Embeddings
        llm, embeddings = get_evaluation_resources()
        
        # 4. Run Ragas evaluate
        logger.info(f"Running evaluation on {len(records)} records using Ragas...")
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings
        )
        
        # 5. Format results
        df = result.to_pandas()
        
        details = []
        for idx, row_data in df.iterrows():
            f_score = row_data.get("faithfulness")
            ar_score = row_data.get("answer_relevancy")
            
            details.append({
                "id": records[idx].get("id"),
                "doc_id": records[idx].get("doc_id"),
                "question": records[idx].get("query"),
                "answer": records[idx].get("response"),
                "contexts": records[idx].get("retrieved_contexts"),
                "scores": {
                    "faithfulness": None if (f_score is None or (isinstance(f_score, float) and math.isnan(f_score))) else float(f_score),
                    "answer_relevancy": None if (ar_score is None or (isinstance(ar_score, float) and math.isnan(ar_score))) else float(ar_score)
                }
            })
            
        overall_scores = {}
        for metric in ["faithfulness", "answer_relevancy"]:
            if metric in df.columns:
                valid_scores = df[metric].dropna()
                overall_scores[metric] = float(valid_scores.mean()) if not valid_scores.empty else None
            
        return {
            "message": "Evaluation completed successfully.",
            "evaluated_count": len(records),
            "overall_scores": overall_scores,
            "details": details
        }
        
    except Exception as e:
        logger.error(f"Ragas evaluation failed: {e}", exc_info=True)
        err_msg = str(e)
        
        # Detect if it's a rate limit or Ragas KeyError: 0 (caused by empty trace on 429)
        is_rate_limit = (
            "rate_limit" in err_msg.lower() or
            "429" in err_msg or
            (isinstance(e, KeyError) and (err_msg == "0" or err_msg == "'0'"))
        )
        
        if is_rate_limit:
            raise HTTPException(
                status_code=429,
                detail="Groq API Rate Limit reached or daily token quota exceeded. Please wait a few minutes for reset or configure a new GROQ_API_KEY in your .env file."
            )
            
        raise HTTPException(status_code=500, detail=f"Ragas evaluation failed: {err_msg}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
