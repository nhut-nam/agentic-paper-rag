import sys
import io
import os
import csv
import time
import json
import re
import requests
import argparse
from dotenv import load_dotenv

# ── Reconfigure stdout/stderr to UTF-8 on Windows ──────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ── Load .env ───────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

BASE_URL = "http://127.0.0.1:8000"

# Tên file PDF ViT trong hệ thống (dùng để tự động tìm doc_id)
VIT_FILENAME_HINT = "ViT"


# ── Tự động lấy doc_id của ViT từ API ───────────────────────────────────────
def get_vit_doc_id() -> str | None:
    """Lấy doc_id của file ViT từ danh sách documents trên API server."""
    try:
        res = requests.get(f"{BASE_URL}/documents", timeout=10)
        if res.status_code == 200:
            docs = res.json()
            for doc in docs:
                path = doc.get("path", "")
                # Tìm doc có path chứa "ViT" (không phân biệt hoa thường)
                if VIT_FILENAME_HINT.lower() in os.path.basename(path).lower():
                    return doc.get("doc_id")
    except Exception as e:
        print(f"[WARN] Không lấy được danh sách documents: {e}")
    return None


# ── Groq Grading ─────────────────────────────────────────────────────────────
def evaluate_answer(question: str, ground_truth: str, agent_answer: str) -> dict:
    """
    Dùng Groq API (llama-3.3-70b-versatile) chấm điểm câu trả lời của Agent.
    Trả về dict {'score': 0-10, 'reason': '...'}.
    """
    if not agent_answer or "TIMEOUT/ERROR" in agent_answer:
        return {"score": 0, "reason": "Agent không đưa ra được câu trả lời hoặc bị lỗi/timeout."}

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Fallback: đọc thẳng từ file .env
        env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "GROQ_API_KEY=" in line and not line.strip().startswith("#"):
                        api_key = line.split("=", 1)[1].strip()
                        break
        except Exception:
            pass

    if not api_key:
        return {"score": 5, "reason": "Không tìm thấy GROQ_API_KEY để chấm điểm."}

    prompt = f"""Bạn là một giám khảo chấm điểm chuyên nghiệp. Hãy so sánh Câu trả lời của Agent với Câu trả lời mẫu (Ground Truth) cho câu hỏi dưới đây và chấm điểm từ 0 đến 10 dựa trên tính chính xác về mặt sự thật, số liệu khoa học và mức độ hoàn thành câu hỏi.

Câu hỏi: "{question}"
Câu trả lời mẫu (Ground Truth): "{ground_truth}"
Câu trả lời của Agent: "{agent_answer}"

Yêu cầu chấm điểm:
- 10: Trả lời hoàn toàn chính xác, đầy đủ các ý chính và số liệu khoa học như câu trả lời mẫu.
- 7-9: Trả lời đúng các ý chính, có thể thiếu một vài chi tiết nhỏ hoặc viết hơi dài dòng nhưng nội dung khoa học chính xác.
- 5-6: Chỉ trả lời đúng một phần nhỏ, hoặc có chứa thông tin chính xác nhưng bị lẫn lộn thông tin sai lệch khác.
- 0-4: Trả lời hoàn toàn sai lệch, lạc đề hoặc trả lời "không tìm thấy thông tin".

Phản hồi duy nhất của bạn phải là một chuỗi JSON hợp lệ có định dạng sau (không thêm bất kỳ từ nào khác ngoài JSON):
{{"score": <điểm số từ 0 đến 10>, "reason": "<giải thích lý do ngắn gọn bằng tiếng Việt>"}}"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }

    try:
        res = requests.post(url, json=payload, headers=headers, timeout=30)
        if res.status_code == 200:
            content = res.json()["choices"][0]["message"]["content"].strip()
            return json.loads(content)
        else:
            return {"score": 5, "reason": f"Lỗi Groq API (Status {res.status_code}): {res.text[:200]}"}
    except Exception as e:
        return {"score": 5, "reason": f"Lỗi thực thi chấm điểm Groq: {e}"}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Benchmark Q&A Agent trên bộ câu hỏi ViT.")
    parser.add_argument("--limit", type=int, default=10, help="Số câu tối đa sẽ chạy (mặc định 10).")
    parser.add_argument("--doc_id", type=str, default=None, help="Override doc_id thủ công nếu cần.")
    args = parser.parse_args()

    # ── Kiểm tra API server ─────────────────────────────────────────────────
    try:
        ping = requests.get(f"{BASE_URL}/documents", timeout=5)
        print(f"[OK] API server đang chạy (status {ping.status_code})")
    except Exception as e:
        print(f"[LỖI] Không kết nối được API server tại {BASE_URL}: {e}")
        print("      Hãy khởi động: venv\\Scripts\\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000")
        return

    # ── Tìm doc_id ViT ─────────────────────────────────────────────────────
    doc_id = args.doc_id
    if not doc_id:
        doc_id = get_vit_doc_id()
    if not doc_id:
        print("[WARN] Không tìm thấy tài liệu ViT trong DB. Dùng doc_id hardcoded dự phòng.")
        doc_id = "90034a5d-b97f-4a34-bacc-ef11780f5c89"
    print(f"[OK] Sử dụng doc_id: {doc_id}")

    # ── Đọc file CSV ─────────────────────────────────────────────────────────
    csv_path = os.path.join(os.path.dirname(__file__), "..", "test", "vit_test.csv")
    if not os.path.exists(csv_path):
        print(f"[LỖI] Không tìm thấy file CSV: {csv_path}")
        return

    questions = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if any(v.strip() for v in row.values()):  # bỏ qua dòng trống
                questions.append(row)

    total_available = len(questions)
    run_limit = min(args.limit, total_available)
    questions = questions[:run_limit]

    print(f"\n=== KHỞI CHẠY BENCHMARK Q&A ===")
    print(f"Tổng câu hỏi có: {total_available} | Sẽ chạy: {run_limit} câu\n")

    correct_count = 0
    results_detail = []
    start_time_all = time.time()

    for idx, q_data in enumerate(questions):
        # Đọc Question/Answer (thử cả chữ hoa lẫn chữ thường)
        question_text = q_data.get("Question") or q_data.get("question") or list(q_data.values())[0]
        ground_truth  = q_data.get("Answer")   or q_data.get("answer")   or list(q_data.values())[1]

        question_text = question_text.strip()
        ground_truth  = ground_truth.strip()

        if not question_text:
            print(f"[{idx+1}] Bỏ qua dòng rỗng.")
            continue

        preview = question_text[:70] + "..." if len(question_text) > 70 else question_text
        print(f"[{idx+1}/{run_limit}] {preview}")

        # ── Gọi Agent API ──────────────────────────────────────────────────
        q_start = time.time()
        agent_answer = "TIMEOUT/ERROR"
        q_latency = 0

        try:
            res = requests.post(
                f"{BASE_URL}/agent/query",
                params={"query": question_text, "doc_id": doc_id},
                timeout=180,      # 3 phút / câu
            )
            q_latency = time.time() - q_start

            if res.status_code == 200:
                agent_answer = res.json().get("final_answer", "").strip() or "TIMEOUT/ERROR"
            else:
                print(f"  [WARN] API status {res.status_code}: {res.text[:200]}")

        except requests.exceptions.Timeout:
            print("  [WARN] Timeout sau 180 giây.")
            q_latency = 180
        except Exception as e:
            print(f"  [WARN] Lỗi kết nối: {e}")

        # ── Chấm điểm Groq ─────────────────────────────────────────────────
        print("  Đang chấm điểm bằng Groq...")
        eval_res = evaluate_answer(question_text, ground_truth, agent_answer)
        score  = eval_res.get("score", 0)
        reason = eval_res.get("reason", "N/A")

        status = "ĐÚNG" if score >= 7 else "SAI"
        if score >= 7:
            correct_count += 1

        print(f"  → {status} (Điểm: {score}/10) | {q_latency:.1f}s")
        print(f"  → Lý do: {reason}")
        print("-" * 60)

        results_detail.append({
            "stt":          idx + 1,
            "question":     question_text,
            "ground_truth": ground_truth,
            "agent_ans":    agent_answer,
            "score":        score,
            "reason":       reason,
            "status":       status,
            "latency":      q_latency,
        })
        time.sleep(0.5)

    # ── Tổng kết ──────────────────────────────────────────────────────────────
    total_duration = time.time() - start_time_all
    accuracy = (correct_count / run_limit * 100) if run_limit > 0 else 0

    print(f"\n{'='*60}")
    print(f"KẾT QUẢ BENCHMARK:")
    print(f"  Số câu đạt (Score >= 7): {correct_count}/{run_limit}")
    print(f"  Accuracy              : {accuracy:.1f}%")
    print(f"  Tổng thời gian        : {total_duration:.1f}s (~{total_duration/60:.1f} phút)")
    print(f"{'='*60}")

    # ── Ghi báo cáo Markdown ──────────────────────────────────────────────────
    report_path = os.path.join(os.path.dirname(__file__), "vit_test_report.md")
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("# BÁO CÁO ĐÁNH GIÁ CHẤT LƯỢNG AGENT — TẬP TỰ LUẬN ViT\n\n")
        rf.write("## 📊 Kết Quả Tổng Quan\n\n")
        rf.write(f"| Chỉ số | Giá trị |\n|---|---|\n")
        rf.write(f"| **Accuracy (Score ≥ 7)** | `{accuracy:.1f}%` ({correct_count}/{run_limit} câu) |\n")
        rf.write(f"| **Tổng thời gian chạy** | `{total_duration:.1f}s` (~{total_duration/60:.1f} phút) |\n")
        rf.write(f"| **Model chấm điểm** | Groq `llama-3.3-70b-versatile` |\n")
        rf.write(f"| **Doc ID** | `{doc_id}` |\n\n")

        rf.write("## 📋 Bảng Điểm Tổng Hợp\n\n")
        rf.write("| STT | Câu hỏi (tóm tắt) | Điểm | Kết quả | Thời gian | Lý do chấm |\n")
        rf.write("|:---:|---|:---:|:---:|:---:|---|\n")
        for r in results_detail:
            emoji = "✅" if r["status"] == "ĐÚNG" else "❌"
            q_short = r["question"][:80] + "..." if len(r["question"]) > 80 else r["question"]
            rf.write(f"| {r['stt']} | {q_short} | `{r['score']}/10` | {emoji} **{r['status']}** | {r['latency']:.1f}s | {r['reason']} |\n")

        rf.write("\n## 🔍 Phân Tích Chi Tiết\n\n")
        for r in results_detail:
            emoji = "✅" if r["status"] == "ĐÚNG" else "❌"
            rf.write(f"### Câu {r['stt']}\n\n")
            rf.write(f"**❓ Câu hỏi:** {r['question']}\n\n")
            rf.write(f"**📖 Ground Truth:**\n> {r['ground_truth']}\n\n")
            rf.write(f"**🤖 Agent Answer:**\n```\n{r['agent_ans']}\n```\n\n")
            rf.write(f"**🏅 Điểm:** `{r['score']}/10` — {emoji} **{r['status']}**\n\n")
            rf.write(f"**💬 Lý do giám khảo:** *{r['reason']}*\n\n")
            rf.write("---\n\n")

    print(f"\n✅ Báo cáo đã lưu tại: {report_path}")


if __name__ == "__main__":
    main()
