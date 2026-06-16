import os
import sys
import io
from dotenv import load_dotenv

# Reconfigure stdout/stderr to use UTF-8 on Windows to avoid UnicodeEncodeErrors
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Load environment variables from .env file
load_dotenv()

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_env():
    """Verify that GROQ_API_KEY is configured."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\n[ERROR] GROQ_API_KEY environment variable is not set!")
        sys.exit(1)
    return api_key

def main():
    print("============================================================")
    print(" RAGAS EVALUATION WITH CHATGROQ (LLAMA 3.1) TEST SCRIPT")
    print("============================================================")
    
    # 1. Verify Groq API Key
    api_key = check_env()
    
    # 2. Install/Import dependencies
    try:
        from langchain_groq import ChatGroq
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, answer_similarity
    except ImportError as e:
        print(f"\n[ERROR] Missing dependencies: {e}")
        sys.exit(1)
        
    print("\n1. Initializing Groq Client & Local Embeddings...")
    # Initialize standard ChatGroq LLM for evaluation
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    
    # Initialize local HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 3. Create dummy test dataset containing English and Vietnamese Q&A
    print("2. Preparing Test Dataset (Multilingual: EN & VI)...")
    test_data = {
        "question": [
            # Test Case 1: English
            "What architecture features are present in DaViT-Small according to Figure 1?",
            # Test Case 2: Vietnamese
            "Mô tả của mô hình DaViT-Small có các thông số C và L bằng bao nhiêu theo tài liệu?"
        ],
        "contexts": [
            [
                "We present the detailed neural network architecture in Figure 1. "
                "The components include Multi-Head Attention, Feed Forward networks, and residual connections. "
                "Refer to the layout diagram here: ../images/fig_model_arch/image.png for structural flows."
            ],
            [
                "This section of the document describes a specific variant of DaViT (Dense Attention Variable Input Transformer), "
                "named DaViT-Small. The model parameters include number of layers C = 96, and segments L = {1, 1, 9, 1}."
            ]
        ],
        "answer": [
            "DaViT-Small features Multi-Head Attention, Feed Forward networks, and residual connections as shown in the Figure 1 block diagram.",
            "Theo tài liệu, mô hình DaViT-Small có các thông số bao gồm số lớp C = 96 và số phân đoạn L = {1, 1, 9, 1}."
        ],
        "ground_truth": [
            "Figure 1 of the document outlines that the DaViT-Small architecture contains Multi-Head Attention, Feed Forward networks, and residual connections.",
            "Mô hình DaViT-Small có các tham số đặc tả cấu hình gồm số lớp C = 96 và phân đoạn L = {1, 1, 9, 1}."
        ]
    }
    
    dataset = Dataset.from_dict(test_data)
    print("   Dataset loaded successfully.")
    
    # 4. Configure metrics with our LLM and Embeddings
    print("3. Configuring Ragas Metrics...")
    
    # We do NOT assign llm and embeddings manually to avoid 'ChatGroq object has no attribute set_run_config'.
    # We pass them to evaluate() below, which wraps and injects them correctly.
    metrics = [faithfulness, answer_relevancy, answer_similarity]
    
    # 5. Run Evaluation
    print("4. Executing Ragas Evaluation (LLM Judge: Llama-3.3-70b)...")
    print("   Please wait, sending queries to Groq LPU...")
    
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings
        )
        
        print("\n============================================================")
        print("                     EVALUATION RESULTS")
        print("============================================================")
        print(f"Overall Metrics Score:")
        print(result)
            
        print("\nDetailed Score per Query:")
        df = result.to_pandas()
        for idx, row in df.iterrows():
            print(f"\n--- Test Case {idx+1} ---")
            q_val = row.get('question', row.get('user_input', ''))
            a_val = row.get('answer', row.get('response', ''))
            print(f"Question: {q_val}")
            print(f"Answer  : {a_val}")
            print(f"Scores  :")
            print(f"  - Faithfulness : {row.get('faithfulness', 0.0):.4f}")
            print(f"  - Relevance    : {row.get('answer_relevancy', 0.0):.4f}")
            print(f"  - Similarity   : {row.get('answer_similarity', row.get('semantic_similarity', 0.0)):.4f}")
            
        print("\n============================================================")
        print("✅ Evaluation completed successfully.")
        
    except Exception as e:
        print(f"\n[ERROR] Ragas evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
