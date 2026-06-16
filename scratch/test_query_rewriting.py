import sys
import io
import uuid
from app.utils.db import DatabaseHandler
from app.agent.analyzer import ComplexityAnalyzer
from app.llm.factory import LLMFactory

# Reconfigure stdout/stderr to use UTF-8 on Windows to avoid UnicodeEncodeErrors
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def main():
    print("=== STARTING QUERY REWRITING TEST ===")
    
    # 1. Initialize Database
    db = DatabaseHandler()
    
    # Generate a unique test session_id
    session_id = f"test-session-{uuid.uuid4().hex[:8]}"
    print(f"Generated Test Session ID: {session_id}")
    
    # Create session chat in DB
    db.create_session_chat(session_id)
    
    # 2. Simulate Q1: User asks a specific question about ViT
    q1 = "Trong kiến trúc Vision Transformer (ViT), một hình ảnh được xử lý như thế nào trước khi đưa vào Transformer encoder?"
    ans1 = (
        "Trước khi đưa hình ảnh vào Transformer encoder trong kiến trúc Vision Transformer (ViT), "
        "hình ảnh 2D được chia thành các mảnh (patches) có kích thước cố định không chồng chéo. "
        "Các mảnh này sau đó được trải phẳng (flatten) thành các vector 1D và được đưa qua một lớp chiếu tuyến tính (linear projection) "
        "để tạo ra các patch embeddings."
    )
    
    print(f"\n[Turn 1] User asks: '{q1}'")
    print(f"[Turn 1] Assistant answers: '{ans1[:100]}...'")
    
    # Insert interaction into query_memory
    memory_id = db.insert_query_memory(
        query=q1,
        response=ans1,
        retrieved_contexts=["Vision Transformer architecture section details..."],
        doc_id="test-doc-id-123",
        session_id=session_id
    )
    print(f"Inserted interaction into query_memory with ID: {memory_id}")
    
    # Generate initial summary using LLM
    llm_provider = LLMFactory.get_provider("ollama")
    
    summary_prompt = f"""
    You are an AI Conversation Summarizer. Your job is to update the summary of a Q&A conversation.
    
    Previous Conversation Summary:
    "No prior conversation."
    
    Latest Turn:
    User Query: "{q1}"
    Assistant Response: "{ans1}"
    
    Provide an updated, concise, high-level summary of the key information discussed so far in this conversation. 
    Keep it in the language of the conversation (preferably Vietnamese or English as used). 
    Do NOT use dialogue or conversational preamble. Write a single dense paragraph.
    
    Updated Summary:"""
    
    print("\nGenerating session summary via LLM...")
    initial_summary = llm_provider.generate(summary_prompt).strip()
    print(f"Generated Summary: '{initial_summary}'")
    
    # Save summary to DB
    db.insert_summary_session(session_id=session_id, summary=initial_summary, memory_id=memory_id)
    print("Saved summary to summary_session table.")
    
    # 3. Simulate Q2: User asks an ambiguous follow-up query
    q2 = "Câu này chọn câu nào: A. pixels, B. 1x1, C. CNN, D. patches"
    print(f"\n[Turn 2] User asks: '{q2}' (Ambiguous query referring to Turn 1)")
    
    # Retrieve the latest summary from DB
    summary_record = db.get_latest_summary_session(session_id)
    retrieved_summary = summary_record["summary"] if summary_record else None
    print(f"Retrieved summary from DB: '{retrieved_summary}'")
    
    # Analyze and rewrite Q2
    analyzer = ComplexityAnalyzer(llm=llm_provider)
    print("\nAnalyzing and rewriting query via ComplexityAnalyzer...")
    analysis = analyzer.analyze(q2, session_summary=retrieved_summary)
    
    print("\n=== TEST RESULTS ===")
    print(f"Original Query: '{q2}'")
    print(f"Rewritten Query: '{analysis.rewritten_query}'")
    print(f"Detected Mode: '{analysis.mode}'")
    print(f"Detected Language: '{analysis.language}'")
    
    # Basic validation
    if analysis.rewritten_query and q2 != analysis.rewritten_query:
        print("\nSUCCESS: Query was successfully rewritten to resolve ambiguities!")
    else:
        print("\nFAILURE: Query was not rewritten or is unchanged.")
        
    # 4. Simulate Turn 3: User asks a completely ambiguous query with no context
    from app.agent.general_agent import GeneralAgent
    q3 = "Tại sao?"
    print(f"\n[Turn 3] User asks: '{q3}' (Completely ambiguous query with no context)")
    
    print("\nAnalyzing and rewriting query via ComplexityAnalyzer...")
    analysis3 = analyzer.analyze(q3, session_summary=None)
    
    print("\n=== TEST RESULTS FOR TURN 3 ===")
    print(f"Original Query: '{q3}'")
    print(f"Detected is_ambiguous: {analysis3.is_ambiguous}")
    print(f"Detected Mode: '{analysis3.mode}'")
    
    if analysis3.is_ambiguous:
        print("\nSUCCESS: Completely ambiguous query correctly detected!")
        # Simulate routing to GeneralAgent
        general_agent = GeneralAgent(llm=llm_provider)
        response_obj = general_agent.run(q3, language="Vietnamese", is_ambiguous=True)
        print(f"GeneralAgent Clarification Response:\n'{response_obj.content}'")
    else:
        print("\nFAILURE: Ambiguity was not detected.")
        
if __name__ == "__main__":
    main()
