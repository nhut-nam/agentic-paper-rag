# Multi-agent RAG

Hệ thống RAG nâng cao kết hợp Multi-Agent (đại lý đa tác vụ) để đọc, phân tích và trả lời câu hỏi về bài báo khoa học. Dự án sử dụng mô hình LLM chạy cục bộ (Ollama) kết hợp LangGraph để điều phối luồng xử lý.

## Kiến trúc hệ thống
Hệ thống được chia thành 2 phần rõ rệt:
1. **Backend (FastAPI)**: Đảm nhận việc xử lý PDF, trích xuất ảnh/bảng biểu, lập chỉ mục vector, lưu lịch sử hội thoại vào database và chạy luồng xử lý Agent.
2. **Frontend (Streamlit)**: Giao diện web gửi/nhận yêu cầu hoàn toàn qua API backend.

### Quy trình xử lý câu hỏi (Agent Loop)
```
[User Query] -> [Complexity Analyzer]
                       |
     +-----------------+-----------------+
     | (Mơ hồ/Không rõ)                  | (Câu hỏi hợp lệ)
     v                                   v
[General Agent]                  [Planner (Lập kế hoạch)]
 (Hỏi lại/Làm rõ)                        |
                                         v
                                  [Research Agent] (Chạy tiếng Anh)
                                  - Hybrid Search (Vector + KeyBERT)
                                  - Web Search
                                  - Vision (Đọc công thức/Bảng)
                                         |
                                         v
                                  [Synthesizer Agent]
                                  - Biên soạn & Lọc nhiễu
                                  - Dịch về tiếng Việt
                                  - Trích dẫn nguồn [1], [2]
```

## Các tính năng chính
- **Đọc PDF & Trích xuất Ảnh**: Tách hình ảnh bảng biểu, công thức toán và chạy OCR để nhúng vào ngữ cảnh tìm kiếm.
- **Tìm kiếm Hybrid Search**: Kết hợp Vector Embedding và trích xuất từ khóa bằng mô hình cục bộ KeyBERT giúp tìm kiếm khớp thuật ngữ chuyên ngành.
- **Bộ nhớ hội thoại động (Session Memory)**: Tự động lưu tóm tắt phiên chat vào bảng `summary_session` trong Postgres. Khi có câu hỏi tiếp theo (ví dụ: *"Câu này chọn câu nào"*), Agent tự load tóm tắt cũ để viết lại thành câu hỏi đầy đủ ngữ cảnh.
- **Hỏi làm rõ câu hỏi mơ hồ**: Tự động phát hiện câu hỏi quá cụt lủn hoặc mơ hồ (khi không có ngữ cảnh cũ) và đưa qua GeneralAgent để phản hồi hỏi lại người dùng làm rõ ý.
- **Đánh giá Ragas**: Tích hợp endpoint đánh giá chất lượng câu trả lời của Agent dựa trên dữ liệu thật lưu trong DB.

---

## Hướng dẫn cài đặt và khởi chạy

### Yêu cầu hệ thống
- Python 3.10+
- Docker (để chạy PostgreSQL + pgvector)
- Ollama (đã cài đặt mô hình `qwen3.5:cloud` hoặc dòng mô hình tương đương)

### Bước 1: Khởi động Database
Khởi động container Postgres có cài sẵn extension pgvector:
```bash
docker compose up -d
```

### Bước 2: Cài đặt thư viện Python
Tạo môi trường ảo và cài đặt dependencies:
```bash
python -m venv venv
venv\Scripts\activate      # Trên Windows
source venv/bin/activate   # Trên Linux/macOS

pip install -r requirements.txt
```

### Bước 3: Cấu hình file `.env`
Tạo file `.env` ở thư mục gốc (nếu chưa có) và điền các thông tin:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=paper_intel
DB_USER=postgres
DB_PASSWORD=postgres
OLLAMA_BASE_URL=http://localhost:11434
# Nếu muốn dùng đánh giá Ragas thì điền thêm key Groq
GROQ_API_KEY=your_groq_api_key
```

### Bước 4: Chạy API Backend (FastAPI)
```bash
$env:PYTHONUNBUFFERED="1"  # Trên Windows PowerShell
venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```
Backend sẽ khởi chạy tại: `http://127.0.0.1:8000`. Database sẽ tự động được chạy lệnh di trú (migration) để tạo các bảng cần thiết trong lần chạy đầu tiên.

### Bước 5: Chạy Giao diện UI (Streamlit)
Mở một terminal mới, kích hoạt môi trường ảo và chạy:
```bash
venv\Scripts\python.exe -m streamlit run app/ui.py
```
Giao diện sẽ tự động mở tại: `http://localhost:8501`.

---

## Kiểm thử & Xác minh tính năng
Bạn có thể chạy trực tiếp script giả lập hội thoại đa lượt để xem tính năng ghi nhớ ngữ cảnh và tự viết lại câu hỏi mơ hồ hoạt động như thế nào:
```bash
$env:PYTHONPATH="."
venv\Scripts\python.exe scratch/test_query_rewriting.py
```
Nếu màn hình in ra kết quả `SUCCESS: Query was successfully rewritten...` và hiển thị câu hỏi mơ hồ được mở rộng đầy đủ ngữ cảnh, hệ thống của bạn đã hoạt động chính xác.

---

## 📊 Kết Quả Đánh Giá Chất Lượng (Evaluation Benchmark)

Dưới đây là kết quả đánh giá chất lượng câu trả lời của Agent trên tập câu hỏi tự luận bài báo ViT (được chấm điểm tự động bởi mô hình Groq `llama-3.3-70b-versatile` dựa trên Ground Truth):

*   **Tỷ lệ chính xác (Accuracy - Score ≥ 7)**: `75.0%` (3/4 câu đúng)
*   **Tổng thời gian chạy**: `596.9s` (~9.9 phút)
*   **Tài liệu thử nghiệm**: ViT Paper (Doc ID: `90034a5d-b97f-4a34-bacc-ef11780f5c89`)

### Bảng kết quả chi tiết:

| STT | Nội dung câu hỏi thử nghiệm | Điểm số | Trạng thái | Lý do / Đánh giá chi tiết |
|:---:|---|:---:|:---:|---|
| 1 | Xác định optimizer và batch size dùng cho pre-training JFT-300M và fine-tuning. | **8/10** | ✅ ĐÚNG | Trả lời chính xác Adam (4096) và SGD (512), chỉ thiếu chi tiết nhỏ về thông số $\beta_1$ và $\beta_2$. |
| 2 | So sánh tài nguyên tính toán (TPUv3-core-days) và độ chính xác của ViT-L/16 và baseline BiT-L trên ImageNet (dựa trên Table 2). | **0/10** | ❌ SAI | Không đọc được bảng dạng Markdown thô nên trả lời không tìm thấy thông tin phù hợp (cần nâng cấp Vision OCR). |
| 3 | Mức độ loại bỏ và cách tiêm thủ công *inductive biases* đặc thù ảnh 2D vào ViT. | **8/10** | ✅ ĐÚNG | Trả lời đúng các ý chính (quá trình trích xuất patch và nội suy 2D cho positional embeddings). |
| 4 | Lý do sử dụng mã hóa vị trí 1D (1D position embeddings) tiêu chuẩn thay vì 2D phức tạp. | **8/10** | ✅ ĐÚNG | Trả lời đúng nguyên nhân từ thực nghiệm của nhóm tác giả (mã hóa 2D không mang lại hiệu năng vượt trội). |
