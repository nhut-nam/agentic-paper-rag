import sys
import io
import os
import csv
import re
import time
import requests
import argparse

# Reconfigure stdout/stderr to use UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_URL = "http://127.0.0.1:8000"
VIT_DOC_ID = "90034a5d-b97f-4a34-bacc-ef11780f5c89"

def extract_answer(agent_response: str) -> str:
    """
    Extracts the multiple choice letter (A, B, C, or D) from the Agent's response.
    """
    if not agent_response:
        return "UNKNOWN"
        
    response = agent_response.strip()
    
    # 1. Check if the response is exactly A, B, C, D (case-insensitive, ignoring punctuation)
    clean_res = re.sub(r"[^a-zA-D]", "", response).upper()
    if clean_res in ["A", "B", "C", "D"]:
        return clean_res
        
    # 2. Look for patterns like "đáp án đúng là C", "chọn đáp án C", "đáp án: C", "chọn C"
    # (case-insensitive)
    patterns = [
        r"(?:đáp án đúng là|đáp án đúng|chọn đáp án|đáp án|chọn)\s*[:\-\s]*([A-D])\b",
        r"(?:correct option is|correct answer is|answer is|choose option|choose|option|choice)\s*[:\-\s]*([A-D])\b"
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
            
    # 3. Look for standalone letters A, B, C, D in brackets or quotes, e.g. (C), [C], "C", 'C'
    bracket_match = re.search(r"[\(\[\'\"]([A-D])[\)\]\'\"]", response, re.IGNORECASE)
    if bracket_match:
        return bracket_match.group(1).upper()

    # 4. Check if there are standalone A, B, C, D letters in the response
    standalone_letters = re.findall(r"\b([A-D])\b", response.upper())
    if len(standalone_letters) == 1:
        return standalone_letters[0]
    
    # 5. If multiple standalone letters, check if the last sentence contains one.
    sentences = re.split(r"[\.\!\?]", response)
    for sentence in reversed(sentences):
        sentence_clean = sentence.strip().upper()
        if sentence_clean:
            for pattern in patterns:
                match = re.search(pattern, sentence_clean)
                if match:
                    return match.group(1)
            letters = re.findall(r"\b([A-D])\b", sentence_clean)
            if len(letters) == 1:
                return letters[0]

    # 6. Fallback: Ask local Ollama LLM directly to extract the choice from the response
    try:
        url = "http://127.0.0.1:11434/api/generate"
        prompt = f"""Phân tích câu trả lời của trợ lý dưới đây và xác định xem đáp án trắc nghiệm được chọn là A, B, C hay D.
        
Câu trả lời của trợ lý:
"{agent_response}"

Chỉ trả về duy nhất 1 chữ cái đại diện cho đáp án được chọn (A, B, C hoặc D). Nếu câu trả lời không chọn đáp án nào hoặc không rõ ràng, trả về UNKNOWN."""
        payload = {
            "model": "qwen2.5:7b-instruct-q4_K_M",
            "prompt": prompt,
            "stream": False
        }
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            ans = res.json().get("response", "").strip().upper()
            if ans in ["A", "B", "C", "D"]:
                return ans
    except Exception:
        pass

    return "UNKNOWN"

def main():
    parser = argparse.ArgumentParser(description="Run ViT multiple choice test benchmark.")
    parser.search_limit = 20  # default to run all
    parser.add_argument("--limit", type=int, default=20, help="Limit the number of questions to run.")
    args = parser.parse_args()
    
    csv_path = "test/vit_test.csv"
    if not os.path.exists(csv_path):
        print(f"Lỗi: Không tìm thấy file test tại {csv_path}")
        return
    
    print(f"=== KHỞI CHẠY ĐÁNH GIÁ CHẤT LƯỢNG AGENT TRÊN TỆP TEST: {csv_path} ===")
    print(f"Cấu hình: doc_id={VIT_DOC_ID} | Giới hạn số câu: {args.limit}")
    
    questions = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)
            
    total_available = len(questions)
    run_limit = min(args.limit, total_available)
    questions = questions[:run_limit]
    print(f"Tổng số câu hỏi được nạp: {total_available}. Sẽ chạy kiểm thử: {run_limit} câu.\n")
    
    correct_count = 0
    results_detail = []
    
    start_time_all = time.time()
    
    for idx, q_data in enumerate(questions):
        stt = q_data.get("STT", str(idx + 1))
        question_text = q_data.get("Câu hỏi")
        ans_a = q_data.get("Đáp án A")
        ans_b = q_data.get("Đáp án B")
        ans_c = q_data.get("Đáp án C")
        ans_d = q_data.get("Đáp án D")
        correct_ans = q_data.get("Đáp án đúng", "").strip().upper()
        explanation = q_data.get("Giải thích")
        
        print(f"[{idx+1}/{run_limit}] Đang xử lý Câu {stt}...")
        
        # Format query prompt for the Agent using "Hãy trả lời" to avoid triggering "analyze" mode
        query_prompt = f"""Hãy trả lời câu hỏi trắc nghiệm dưới đây liên quan đến tài liệu Vision Transformer và chọn đáp án đúng nhất (A, B, C hoặc D).
        
Câu hỏi: {question_text}
A. {ans_a}
B. {ans_b}
C. {ans_c}
D. {ans_d}

Chỉ trả về duy nhất một chữ cái đại diện cho đáp án đúng (A, B, C hoặc D). Không thêm bất kỳ thông tin hay lời giải thích nào khác."""

        # Send request to Agent API
        url = f"{BASE_URL}/agent/query"
        payload = {
            "query": query_prompt,
            "doc_id": VIT_DOC_ID
        }
        
        q_start = time.time()
        agent_answer = "TIMEOUT/ERROR"
        extracted_ans = "UNKNOWN"
        status = "SAI"
        
        try:
            # Set timeout to 300 seconds because ReAct loops with local models and VLM can take time
            res = requests.post(url, params=payload, timeout=300)
            q_latency = time.time() - q_start
            
            if res.status_code == 200:
                res_data = res.json()
                agent_answer = res_data.get("final_answer", "").strip()
                extracted_ans = extract_answer(agent_answer)
                
                if extracted_ans == correct_ans:
                    correct_count += 1
                    status = "ĐÚNG"
                print(f"  Thời gian phản hồi: {q_latency:.2f}s | Trích xuất đáp án: {extracted_ans} | Đáp án đúng: {correct_ans} | Kết quả: {status}")
            else:
                print(f"  Lỗi API (Status {res.status_code}): {res.text}")
                q_latency = 0
        except requests.exceptions.Timeout:
            print("  Lỗi: Yêu cầu bị Timeout (quá 300 giây).")
            q_latency = 300
        except Exception as e:
            print(f"  Lỗi kết nối tới API server: {e}")
            q_latency = 0
            
        results_detail.append({
            "stt": stt,
            "question": question_text,
            "correct_ans": correct_ans,
            "agent_ans": extracted_ans,
            "agent_raw": agent_answer,
            "status": status,
            "latency": q_latency,
            "explanation": explanation
        })
        time.sleep(1.0) # Short cooldown between requests
        
    end_time_all = time.time()
    total_duration = end_time_all - start_time_all
    
    accuracy = (correct_count / run_limit) * 100 if run_limit > 0 else 0
    print(f"\n==========================================")
    print(f"KẾT QUẢ KIỂM THỬ:")
    print(f"- Số câu đúng: {correct_count}/{run_limit}")
    print(f"- Độ chính xác (Accuracy): {accuracy:.2f}%")
    print(f"- Tổng thời gian thực thi: {total_duration:.2f} giây")
    print(f"==========================================")
    
    # 4. Xuất báo cáo Markdown
    report_path = "scratch/vit_test_report.md"
    print(f"Đang ghi báo cáo chi tiết vào {report_path}...")
    
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write(f"# BÁO CÁO ĐÁNH GIÁ ĐỘ CHÍNH XÁC CỦA AGENT TRÊN TẬP CÂU HỎI VIT\n\n")
        rf.write(f"## 📊 Kết Quả Tổng Quan\n")
        rf.write(f"- **Độ chính xác (Accuracy):** `{accuracy:.2f}%` ({correct_count}/{run_limit} câu đúng)\n")
        rf.write(f"- **Tổng thời gian chạy:** `{total_duration:.2f} giây` (~{total_duration/60:.2f} phút)\n")
        rf.write(f"- **Thời gian trung bình mỗi câu:** `{total_duration/run_limit:.2f} giây` đối với câu hỏi ReAct\n")
        rf.write(f"- **Tài liệu tham chiếu:** `ViT.pdf` (ID: `{VIT_DOC_ID}`)\n\n")
        
        rf.write(f"## 📋 Chi Tiết Kết Quả Từng Câu Hỏi\n\n")
        rf.write(f"| STT | Câu hỏi | Đáp án đúng | Đáp án Agent | Kết quả | Thời gian (s) |\n")
        rf.write(f"|---|---|:---:|:---:|:---:|:---:|\n")
        
        for r in results_detail:
            status_emoji = "✅" if r["status"] == "ĐÚNG" else "❌"
            rf.write(f"| {r['stt']} | {r['question']} | `{r['correct_ans']}` | `{r['agent_ans']}` | {status_emoji} **{r['status']}** | {r['latency']:.1f} |\n")
            
        rf.write(f"\n## 🔍 Phân Tích Lời Giải Thích & Đáp Án Thô Từ Agent\n\n")
        for r in results_detail:
            status_emoji = "✅" if r["status"] == "ĐÚNG" else "❌"
            rf.write(f"### Câu {r['stt']}: {r['question']}\n")
            rf.write(f"- **Đáp án đúng trong CSV:** `{r['correct_ans']}` (Giải thích: *{r['explanation']}*)\n")
            rf.write(f"- **Đáp án Agent trích xuất:** `{r['agent_ans']}`\n")
            rf.write(f"- **Kết quả:** {status_emoji} **{r['status']}**\n")
            rf.write(f"- **Câu trả lời thô của Agent:**\n")
            rf.write(f"  ```text\n  {r['agent_raw']}\n  ```\n")
            rf.write(f"---\n\n")
            
    print("Ghi báo cáo hoàn tất!")

if __name__ == "__main__":
    main()
