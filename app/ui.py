import streamlit as st
import os
import sys
import time
import requests
import uuid
from dotenv import load_dotenv

# Reconfigure stdout to use UTF-8 on Windows to prevent UnicodeEncodeErrors
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Set Page Config for premium look
st.set_page_config(
    page_title="Paper Intelligent AI Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_URL = "http://127.0.0.1:8000"

# Custom Styling (Vanilla CSS with Dark Mode and Sleek Gradients)
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Background & Container */
    .stApp {
        background-color: #0d0f12;
        color: #e2e8f0;
    }
    
    /* Premium Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #12161a !important;
        border-right: 1px solid #1f2937;
    }
    
    /* Sleek Title */
    .dashboard-title {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #1d4ed8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-section {
        background-color: #171d24;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #2d3748;
        margin-bottom: 1rem;
    }
    
    /* Message styling */
    .chat-bubble {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        border: 1px solid #1f2937;
    }
    .user-bubble {
        background-color: #1e293b;
        border-left: 4px solid #3b82f6;
    }
    .assistant-bubble {
        background-color: #0f172a;
        border-left: 4px solid #10b981;
    }
    
    /* Ingest status badges */
    .badge-completed {
        background-color: #065f46;
        color: #34d399;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-pending {
        background-color: #78350f;
        color: #fbbf24;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Scan project directories for local PDF files
def scan_project_pdfs():
    pdfs = []
    # 1. Scan root directory
    for file in os.listdir("."):
        if file.endswith(".pdf"):
            pdfs.append(os.path.abspath(file))
            
    # 2. Scan storage/uploads directory
    uploads_dir = os.path.join("storage", "uploads")
    if os.path.exists(uploads_dir):
        for file in os.listdir(uploads_dir):
            if file.endswith(".pdf"):
                pdfs.append(os.path.abspath(os.path.join(uploads_dir, file)))
                
    return sorted(list(set(pdfs)))

# Call API to fetch all documents
def api_get_documents():
    try:
        res = requests.get(f"{BASE_URL}/documents")
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        st.error(f"Không thể kết nối tới API Server tại {BASE_URL}. Vui lòng khởi động API server trước.")
    return []

# Call API to upload local PDF file
def api_upload_pdf(pdf_path: str):
    try:
        with open(pdf_path, "rb") as f:
            files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
            res = requests.post(f"{BASE_URL}/upload-pdf", files=files)
            if res.status_code == 200:
                return res.json()
    except Exception as e:
        st.error(f"Lỗi upload tệp tin: {e}")
    return None

# Call API to trigger document ingestion
def api_trigger_ingest(doc_id: str):
    try:
        res = requests.post(f"{BASE_URL}/ingest/{doc_id}")
        return res.json()
    except Exception as e:
        st.error(f"Lỗi kích hoạt ingest: {e}")
    return None

# Call API to get document status
def api_get_status(doc_id: str):
    try:
        res = requests.get(f"{BASE_URL}/status/{doc_id}")
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None

# Call API to run Ragas evaluation
def api_run_ragas_eval(memory_id: int):
    try:
        res = requests.post(f"{BASE_URL}/evaluate-memory", params={"memory_id": memory_id})
        if res.status_code == 200:
            results = res.json()
            return results.get("overall_scores"), None
        elif res.status_code == 429:
            return None, "Hạn mức API Groq (Rate Limit 429) của bạn đã hết. Hãy đợi vài phút hoặc đổi API key khác."
        else:
            return None, res.json().get("detail", "Lỗi đánh giá Ragas.")
    except Exception as e:
        return None, str(e)

# Initialize Streamlit Session States
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_doc_ids" not in st.session_state:
    st.session_state.active_doc_ids = []
if "active_doc_names" not in st.session_state:
    st.session_state.active_doc_names = []
if "eval_scores" not in st.session_state:
    st.session_state.eval_scores = {}
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "session_summary" not in st.session_state:
    st.session_state.session_summary = None

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown('<div class="dashboard-title">📚 Paper Intel AI</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #6b7280; font-size: 0.9rem;">Khung Chat Agent & Pipeline Đánh Giá Ragas</p>', unsafe_allow_html=True)
    st.write("---")
    
    # Section 1: Active Session Context
    st.markdown("### 🧠 Ngữ Cảnh Session Hiện Tại")
    if st.session_state.session_id:
        st.markdown(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
        
        summary = st.session_state.session_summary
        if summary:
            st.markdown("**Tóm tắt phiên chat (Session Summary):**")
            st.info(summary)
        else:
            st.markdown("*Chưa có tóm tắt hội thoại.*")
    else:
        st.markdown("*Chưa có session hoạt động.*")
        
    if st.button("🔄 Khởi Tạo Session Mới", use_container_width=True):
        st.session_state.messages = []
        st.session_state.eval_scores = {}
        st.session_state.session_id = None
        st.session_state.session_summary = None
        st.success("Đã tạo phiên hội thoại mới!")
        time.sleep(1)
        st.rerun()
        
    st.write("---")
    
    # Section 2: Upload New PDF
    st.markdown("### 📤 Tải Lên PDF Mới")
    uploaded_file = st.file_uploader("Chọn tệp PDF", type=["pdf"])
    if uploaded_file is not None:
        # Save file temporarily to disk to upload via requests files
        os.makedirs("storage/uploads", exist_ok=True)
        temp_path = os.path.abspath(os.path.join("storage", "uploads", uploaded_file.name))
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        if st.button("Register & Upload lên API Server", use_container_width=True):
            with st.spinner("Đang gửi tệp tới API Server..."):
                res_upload = api_upload_pdf(temp_path)
                if res_upload and res_upload.get("status") == "uploaded":
                    st.success("Tải tệp lên API server thành công!")
                    time.sleep(1)
                    st.rerun("i tệp lên API server thành công!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Có lỗi xảy ra khi tải tệp lên API server.")

    # Section 2: Global/Local PDFs Selection
    st.markdown("### 📂 Thư Viện Tệp Tin (Global PDFs)")
    local_pdfs = scan_project_pdfs()
    
    # Fetch all documents currently registered in DB via API
    registered_docs = api_get_documents()
    path_to_doc = {}
    for doc in registered_docs:
        abs_path = os.path.abspath(doc["path"])
        if abs_path in path_to_doc:
            existing_status = path_to_doc[abs_path].get("status")
            new_status = doc.get("status")
            if existing_status not in ["completed", "chunked"] and new_status in ["completed", "chunked"]:
                path_to_doc[abs_path] = doc
        else:
            path_to_doc[abs_path] = doc

    if not local_pdfs:
        st.warning("Không tìm thấy tệp PDF nào trong dự án.")
    else:
        pdf_options = {}
        for path in local_pdfs:
            filename = os.path.basename(path)
            db_doc = path_to_doc.get(path)
            
            if db_doc:
                status_label = f"[{db_doc['status']}] {filename}"
            else:
                status_label = f"[Chưa đăng ký] {filename}"
            pdf_options[path] = status_label
            
        if "active_pdf_paths" not in st.session_state:
            st.session_state.active_pdf_paths = []

        selected_pdf_paths = st.multiselect(
            "Chọn tài liệu hoạt động",
            options=list(pdf_options.keys()),
            format_func=lambda x: pdf_options[x],
            default=st.session_state.active_pdf_paths
        )
        st.session_state.active_pdf_paths = selected_pdf_paths
        
        if selected_pdf_paths:
            active_ids = []
            active_names = []
            unregistered_paths = []
            pending_ingest_docs = []
            
            for path in selected_pdf_paths:
                db_doc = path_to_doc.get(path)
                if not db_doc:
                    unregistered_paths.append(path)
                else:
                    active_ids.append(db_doc["doc_id"])
                    active_names.append(os.path.basename(db_doc["path"]))
                    if db_doc["status"] not in ["chunked", "completed"]:
                        pending_ingest_docs.append(db_doc)
                        
            st.session_state.active_doc_ids = active_ids
            st.session_state.active_doc_names = active_names
            
            if unregistered_paths:
                st.warning(f"Có {len(unregistered_paths)} tệp chưa đăng ký trên API server.")
                if st.button("Đăng ký tất cả các tệp này", use_container_width=True):
                    with st.spinner("Đang đăng ký các tệp tin..."):
                        for u_path in unregistered_paths:
                            api_upload_pdf(u_path)
                        st.success("Đăng ký thành công!")
                        time.sleep(0.5)
                        st.rerun()
            
            st.markdown("**Trạng thái các tài liệu đã chọn:**")
            for path in selected_pdf_paths:
                db_doc = path_to_doc.get(path)
                if db_doc:
                    status = db_doc["status"]
                    filename = os.path.basename(path)
                    if status in ["chunked", "completed"]:
                        st.markdown(f"✅ `{filename}`: <span class='badge-completed'>Đã Ingest</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"⏳ `{filename}`: <span class='badge-pending'>{status}</span>", unsafe_allow_html=True)
            
            if pending_ingest_docs:
                if st.button("🚀 Ingest các tài liệu chưa xử lý", use_container_width=True):
                    status_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    
                    for doc in pending_ingest_docs:
                        api_trigger_ingest(doc["doc_id"])
                        
                    for percent in range(1, 101):
                        all_done = True
                        statuses = []
                        for doc in pending_ingest_docs:
                            latest_doc = api_get_status(doc["doc_id"])
                            current_status = latest_doc["status"] if latest_doc else "unknown"
                            statuses.append(f"{os.path.basename(doc['path'])}: {current_status}")
                            if current_status not in ["chunked", "completed"]:
                                all_done = False
                                
                        status_placeholder.info("Đang xử lý Ingest...\n" + "\n".join(statuses))
                        progress_bar.progress(percent)
                        
                        if all_done:
                            st.success("Tất cả tài liệu đã Ingest hoàn tất!")
                            time.sleep(1)
                            st.rerun()
                            break
                        time.sleep(1.5)
                    else:
                        st.warning("Quá trình Ingest đang chạy ngầm. Vui lòng F5 trang sau ít phút.")
                        st.rerun()
            else:
                if st.button("🔄 Ingest lại tất cả tài liệu", use_container_width=True):
                    for path in selected_pdf_paths:
                        db_doc = path_to_doc.get(path)
                        if db_doc:
                            api_trigger_ingest(db_doc["doc_id"])
                    st.success("Đã yêu cầu Ingest lại!")
                    time.sleep(0.5)
                    st.rerun()
        else:
            st.session_state.active_doc_ids = []
            st.session_state.active_doc_names = []
                                
    st.write("---")
    if st.button("🗑️ Xóa Lịch Sử Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.eval_scores = {}
        st.rerun()

# ----------------- MAIN AREA -----------------
st.markdown('<div class="dashboard-title">💬 Trò Chuyện Với Tài Liệu Trí Tuệ AI</div>', unsafe_allow_html=True)

# Document header badge
if st.session_state.get("active_doc_ids"):
    names_str = ", ".join([f"`{name}`" for name in st.session_state.active_doc_names])
    st.info(f"📍 **Đang kết nối ({len(st.session_state.active_doc_ids)} tài liệu):** {names_str}")
else:
    st.warning("⚠️ Chưa chọn tài liệu. Agent sẽ trả lời dựa trên kiến thức toàn cục hoặc web search.")

# Display Chat History
for message in st.session_state.messages:
    role = message["role"]
    with st.chat_message(role):
        st.write(message["content"])
        
        # Render additional attributes for assistant answers
        if role == "assistant":
            memory_id = message.get("memory_id")
            retrieved_docs = message.get("retrieved_docs", [])
            
            # 1. Expandable contexts sources
            if retrieved_docs:
                with st.expander("🔍 Xem nguồn văn bản gốc thu hồi (Retrieved Chunks)"):
                    for i, chunk in enumerate(retrieved_docs):
                        st.markdown(f"**Đoạn {i+1}:**")
                        st.text_area("", value=chunk, height=120, disabled=True, key=f"chunk_{memory_id}_{i}")
            
            # 2. Ragas evaluation metrics
            if memory_id:
                st.markdown(f"<p style='color: #6b7280; font-size: 0.85rem; margin-top: 0.5rem;'>Memory Interaction ID: #{memory_id}</p>", unsafe_allow_html=True)
                
                eval_key = f"eval_{memory_id}"
                if eval_key not in st.session_state.eval_scores:
                    st.session_state.eval_scores[eval_key] = None
                    
                scores = st.session_state.eval_scores[eval_key]
                
                if scores is None:
                    if st.button("📊 Đánh giá chất lượng (Ragas)", key=f"btn_{memory_id}"):
                        with st.spinner("Đang gửi yêu cầu đánh giá tới API Server..."):
                            res_scores, err = api_run_ragas_eval(memory_id)
                            if err:
                                st.error(err)
                            else:
                                st.session_state.eval_scores[eval_key] = res_scores
                                st.rerun()
                else:
                    st.markdown("##### Kết quả đánh giá Ragas:")
                    col1, col2 = st.columns(2)
                    col1.metric(
                        label="Faithfulness (Độ trung thực)", 
                        value=f"{scores['faithfulness']:.2f}" if scores['faithfulness'] is not None else "N/A"
                    )
                    col2.metric(
                        label="Answer Relevancy (Độ liên quan)", 
                        value=f"{scores['answer_relevancy']:.2f}" if scores['answer_relevancy'] is not None else "N/A"
                    )

# Chat Input control
if user_query := st.chat_input("Nhập câu hỏi của bạn về tài liệu ở đây..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.rerun()

# Processing the latest user query
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    query_to_run = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        with st.spinner("Agent đang suy nghĩ và thực thi các công cụ qua API Server..."):
            try:
                # Call agent query API endpoint
                payload = {"query": query_to_run}
                if st.session_state.get("active_doc_ids"):
                    payload["doc_ids"] = ",".join(st.session_state.active_doc_ids)
                    
                res = requests.post(f"{BASE_URL}/agent/query", params=payload)
                
                if res.status_code == 200:
                    res_data = res.json()
                    final_answer = res_data.get("final_answer", "Không có câu trả lời.")
                    memory_id = res_data.get("memory_id")
                    retrieved_docs = res_data.get("retrieved_docs", [])
                    
                    st.write(final_answer)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_answer,
                        "memory_id": memory_id,
                        "retrieved_docs": retrieved_docs
                    })
                    st.rerun()
                else:
                    st.error(f"API Server trả về lỗi {res.status_code}: {res.text}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Thất bại khi kết nối server: {res.text}"
                    })
            except Exception as e:
                st.error(f"Kết nối tới API Server thất bại: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Không thể kết nối tới API Server tại {BASE_URL}."
                })
