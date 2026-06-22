"""
test_low_score_search.py
========================
Kiểm thử cơ chế chuyển hướng sang Web Search khi score tài liệu thấp (< 0.5).

Câu hỏi hoàn toàn không liên quan đến ViT paper nên retrieve_tool sẽ trả
về score thấp và Agent phải tự động gọi web_search_tool.

Script này gọi thẳng vào API server đang chạy (port 8000) thay vì
import agent trực tiếp để tránh việc bị treo do Ollama blocking.
"""

import sys
import io
import os
import time
import requests

# ── Reconfigure stdout/stderr to UTF-8 on Windows ──────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_URL = "http://127.0.0.1:8000"
# ViT paper doc_id đang có trong DB (đổi nếu cần)
VIT_DOC_ID = "90034a5d-b97f-4a34-bacc-ef11780f5c89"


def main():
    print("=== KIỂM THỬ WEB SEARCH KHI ĐỘ LIÊN QUAN TÀI LIỆU THẤP ===\n")

    # Câu hỏi hoàn toàn KHÔNG liên quan đến ViT → score < 0.5
    query = "Who is the current Prime Minister of Vietnam?"
    print(f"Câu hỏi : '{query}'")
    print(f"Doc ID   : {VIT_DOC_ID}")
    print("-" * 60)

    # ── Kiểm tra API server đang chạy ───────────────────────────────────────
    try:
        ping = requests.get(f"{BASE_URL}/documents", timeout=5)
        print(f"[OK] API server phản hồi (status {ping.status_code})\n")
    except Exception as e:
        print(f"[LỖI] Không kết nối được API server tại {BASE_URL}: {e}")
        print("      Hãy đảm bảo server đang chạy bằng lệnh:")
        print("      venv\\Scripts\\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000")
        return

    # ── Gọi Agent ────────────────────────────────────────────────────────────
    url = f"{BASE_URL}/agent/query"
    params = {
        "query": query,
        "doc_id": VIT_DOC_ID,
    }

    print("Đang gửi yêu cầu tới Agent (timeout 120s)...")
    start = time.time()
    try:
        res = requests.post(url, params=params, timeout=120)
        elapsed = time.time() - start
    except requests.exceptions.Timeout:
        print("[LỖI] Yêu cầu bị Timeout sau 120 giây.")
        return
    except Exception as e:
        print(f"[LỖI] Kết nối thất bại: {e}")
        return

    if res.status_code != 200:
        print(f"[LỖI] API trả về status {res.status_code}:\n{res.text}")
        return

    data = res.json()
    final_answer = data.get("final_answer", "(không có final_answer)")
    tasks = data.get("tasks", [])

    # ── Gom thought process và context_used từ các task ─────────────────────
    all_context_used: list[str] = []
    all_thought_parts: list[str] = []
    for t in tasks:
        all_context_used.extend(t.get("context_used", []))
        all_thought_parts.append(t.get("thought_process", ""))

    # ── In kết quả ───────────────────────────────────────────────────────────
    print(f"\n=== KẾT QUẢ (sau {elapsed:.1f}s) ===")
    print(f"Final Answer:\n{final_answer}\n")
    print(f"Context Used:\n" + "\n".join(f"  - {c}" for c in all_context_used))
    print("-" * 60)

    # ── Đánh giá ─────────────────────────────────────────────────────────────
    web_called     = any("web_search_tool" in c for c in all_context_used)
    retrieve_called = any("retrieve_tool"  in c for c in all_context_used)

    print("\n=== ĐÁNH GIÁ CHẤT LƯỢNG TEST ===")
    if retrieve_called:
        print("✅ 1. retrieve_tool đã được gọi trước.")
    else:
        print("❌ 1. retrieve_tool KHÔNG được gọi.")

    if web_called:
        print("✅ 2. Agent tự động chuyển sang web_search_tool khi score thấp.")
    else:
        print("❌ 2. Agent KHÔNG chuyển sang web_search_tool.")

    print(f"\n⏱  Thời gian phản hồi: {elapsed:.1f}s")
    print("=== HOÀN TẤT KIỂM THỬ ===")


if __name__ == "__main__":
    main()
