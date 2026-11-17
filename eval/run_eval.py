"""
Benchmark the multi-agent RAG on the ViT paper (eval/vit_qa.json, 28 questions).

For every question it records:
  * fact recall   - deterministic: how many required facts (numbers/terms) appear in the answer
  * correct       - all facts present (or, for unanswerable questions, the agent abstained)
  * evidence hit  - did retrieval bring back a chunk that contains the answer? (separates
                    retrieval failures from reasoning/synthesis failures)
  * LLM judge     - optional Groq score 0-10 vs ground truth; API errors are reported as
                    errors, never as a fake score
  * latency

Usage (backend must be running: uvicorn app.api.main:app --port 8000):
    python -m eval.run_eval --label before
    python -m eval.run_eval --label after --only B01,B02
    python -m eval.compare eval/results/before.json eval/results/after.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from eval.scoring import abstained, evidence_hit, fact_hits

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
load_dotenv(ROOT / ".env")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def find_doc_id(base_url: str, hint: str) -> str | None:
    try:
        docs = requests.get(f"{base_url}/documents", timeout=10).json()
    except Exception as e:
        print(f"[WARN] cannot list documents: {e}")
        return None
    for d in docs:
        if hint.lower() in os.path.basename(d.get("path", "")).lower():
            return d.get("doc_id")
    return None


def judge(question: str, ground_truth: str, answer: str, model: str) -> dict:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return {"score": None, "error": "GROQ_API_KEY missing"}
    prompt = (
        "You grade answers about a research paper. Compare the agent answer with the reference answer "
        "and give a score 0-10 for factual correctness and completeness. Numbers must match exactly. "
        "If the reference says the paper does not contain the information, a correct answer must say so "
        "and must not invent numbers.\n\n"
        f"Question: {question}\nReference answer: {ground_truth}\nAgent answer: {answer}\n\n"
        'Reply with JSON only: {"score": <0-10>, "reason": "<short reason>"}'
    )
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "temperature": 0, "response_format": {"type": "json_object"},
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        if r.status_code != 200:
            return {"score": None, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
        data = json.loads(r.json()["choices"][0]["message"]["content"])
        return {"score": int(data.get("score")), "reason": data.get("reason", "")}
    except Exception as e:
        return {"score": None, "error": f"{type(e).__name__}: {e}"}


def run_item(item: dict, base_url: str, doc_id: str, timeout: int) -> dict:
    t0 = time.time()
    answer, retrieved, error = "", [], None
    try:
        r = requests.post(f"{base_url}/agent/query", params={"query": item["question"], "doc_id": doc_id},
                          timeout=timeout)
        if r.status_code == 200:
            body = r.json()
            answer = (body.get("final_answer") or "").strip()
            retrieved = body.get("retrieved_docs") or []
        else:
            error = f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    latency = round(time.time() - t0, 1)

    if item.get("expect_abstain"):
        hits, recall, correct = [], None, bool(answer) and abstained(answer)
    else:
        hits = fact_hits(answer, item["key_facts"])
        recall = round(sum(hits) / len(hits), 3)
        correct = all(hits)
    return {
        "id": item["id"], "category": item["category"], "question": item["question"],
        "ground_truth": item["ground_truth"], "answer": answer, "error": error,
        "fact_hits": hits, "fact_recall": recall, "correct": correct,
        "evidence_hit": evidence_hit(retrieved, item["evidence"]) if item.get("evidence") else None,
        "retrieved_count": len(retrieved), "latency_s": latency,
    }


def summarize(rows: list[dict]) -> dict:
    def pct(xs):
        xs = [x for x in xs if x is not None]
        return round(100 * sum(xs) / len(xs), 1) if xs else None

    cats: dict[str, list] = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)
    judged = [r["judge"]["score"] for r in rows if r.get("judge", {}).get("score") is not None]
    return {
        "questions": len(rows),
        "accuracy": pct([r["correct"] for r in rows]),
        "accuracy_by_category": {c: f"{sum(r['correct'] for r in rs)}/{len(rs)}" for c, rs in cats.items()},
        "avg_fact_recall": pct([r["fact_recall"] for r in rows if r["fact_recall"] is not None]),
        "evidence_hit_rate": pct([r["evidence_hit"] for r in rows]),
        "judge_pass_rate_ge7": pct([s >= 7 for s in judged]) if judged else None,
        "judge_avg": round(sum(judged) / len(judged), 2) if judged else None,
        "judge_errors": sum(1 for r in rows if r.get("judge", {}).get("error")),
        "request_errors": sum(1 for r in rows if r["error"]),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / max(1, len(rows)), 1),
    }


def to_markdown(label: str, s: dict, rows: list[dict]) -> str:
    out = [f"# RAG eval - {label} ({datetime.now():%Y-%m-%d %H:%M})", "",
           f"Accuracy (all key facts present / correct abstain): **{s['accuracy']}%** of {s['questions']}  ",
           f"By category: {', '.join(f'{k} {v}' for k, v in s['accuracy_by_category'].items())}  ",
           f"Avg fact recall: {s['avg_fact_recall']}% | Evidence hit rate (retrieval): {s['evidence_hit_rate']}%  ",
           f"LLM judge >=7: {s['judge_pass_rate_ge7']}% (avg {s['judge_avg']}, errors {s['judge_errors']}) | "
           f"Avg latency: {s['avg_latency_s']}s", "",
           "| ID | Cat | Correct | Facts | Retrieved evidence | Judge | Latency |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        facts = "abstain" if r["fact_recall"] is None else f"{sum(r['fact_hits'])}/{len(r['fact_hits'])}"
        ev = {True: "yes", False: "NO", None: "-"}[r["evidence_hit"]]
        j = r.get("judge", {})
        js = j.get("score") if j.get("score") is not None else ("err" if j.get("error") else "-")
        out.append(f"| {r['id']} | {r['category']} | {'✅' if r['correct'] else '❌'} | {facts} | {ev} | {js} | {r['latency_s']}s |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--dataset", default=str(Path(__file__).with_name("vit_qa.json")))
    ap.add_argument("--doc-id", default=None)
    ap.add_argument("--label", default=datetime.now().strftime("run-%Y%m%d-%H%M"))
    ap.add_argument("--only", default="", help="comma-separated ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-model", default="llama-3.3-70b-versatile")
    args = ap.parse_args()

    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    items = data["items"]
    if args.only:
        keep = set(args.only.split(","))
        items = [i for i in items if i["id"] in keep]
    if args.limit:
        items = items[: args.limit]

    doc_id = args.doc_id or find_doc_id(args.base_url, data.get("document_hint", ""))
    if not doc_id:
        sys.exit("Could not find the document. Pass --doc-id.")
    print(f"doc_id={doc_id} | {len(items)} questions | label={args.label}")

    rows = []
    for n, item in enumerate(items, 1):
        print(f"[{n}/{len(items)}] {item['id']} {item['question'][:70]}...", flush=True)
        row = run_item(item, args.base_url, doc_id, args.timeout)
        if not args.no_judge:
            row["judge"] = judge(item["question"], item["ground_truth"], row["answer"], args.judge_model)
        print(f"    correct={row['correct']} recall={row['fact_recall']} evidence={row['evidence_hit']} "
              f"judge={row.get('judge', {}).get('score')} ({row['latency_s']}s)", flush=True)
        rows.append(row)

    summary = summarize(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"label": args.label, "doc_id": doc_id, "time": datetime.now().isoformat(timespec="seconds"),
               "summary": summary, "results": rows}
    (RESULTS_DIR / f"{args.label}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = to_markdown(args.label, summary, rows)
    (RESULTS_DIR / f"{args.label}.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
