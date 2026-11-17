"""Compare two eval runs: python -m eval.compare eval/results/before.json eval/results/after.json"""
import json
import sys
from pathlib import Path


def main(a_path: str, b_path: str) -> None:
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (a_path, b_path))
    sa, sb = a["summary"], b["summary"]
    print(f"| Metric | {a['label']} | {b['label']} |\n|---|---|---|")
    for k in ("accuracy", "avg_fact_recall", "evidence_hit_rate", "judge_pass_rate_ge7", "judge_avg", "avg_latency_s"):
        print(f"| {k} | {sa.get(k)} | {sb.get(k)} |")
    cats = sorted(set(sa["accuracy_by_category"]) | set(sb["accuracy_by_category"]))
    for c in cats:
        print(f"| {c} | {sa['accuracy_by_category'].get(c, '-')} | {sb['accuracy_by_category'].get(c, '-')} |")
    ra = {r["id"]: r for r in a["results"]}
    changed = [(r["id"], ra[r["id"]]["correct"], r["correct"]) for r in b["results"]
               if r["id"] in ra and ra[r["id"]]["correct"] != r["correct"]]
    if changed:
        print("\nChanged questions:")
        for i, x, y in changed:
            print(f"- {i}: {'✅' if x else '❌'} -> {'✅' if y else '❌'}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
