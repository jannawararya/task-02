"""Run tests/eval_questions.csv through the chatbot and write docs/eval_results.csv.

Usage:
  python evaluate.py                    full run (retrieval + generation)
  python evaluate.py --retrieval-only   fast: only checks which pages are retrieved
  python evaluate.py --min-score 0.35   try a different abstention threshold

AUTOMATIC columns  : retrieved sources, citation_hit, abstained, abstention_ok
MANUAL columns     : result (Correct / Partially correct / Incorrect), notes
Answer correctness cannot be judged automatically, so grade `result` yourself
by reading chatbot_answer against expected_answer and the cited page.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from rag_core import MIN_SCORE, TOP_K, Generator, Retriever, answer_question

ROOT = Path(__file__).resolve().parent
QUESTIONS = ROOT / "tests" / "eval_questions.csv"
DEFAULT_OUT = ROOT / "docs" / "eval_results.csv"
SUMMARY = ROOT / "docs" / "eval_summary.json"


def citation_hit(expected_pages: str, expected_source: str, sources: list[dict]) -> str:
    expected = {int(p) for p in expected_pages.split(";") if p.strip()}
    got = {s["page"] for s in sources
           if not expected_source or expected_source.lower() in s["source"].lower()}
    if expected <= got:
        return "yes"
    return "partial" if expected & got else "no"


def main(args) -> None:
    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        existing = pd.read_csv(out_path, dtype=str, keep_default_na=False)
        if "result" in existing and existing["result"].str.strip().ne("").any():
            raise SystemExit(f"{out_path} already contains manual grades. "
                             "Use --out for a new file or --force to overwrite.")

    df = pd.read_csv(QUESTIONS, dtype=str, keep_default_na=False)
    retriever = Retriever()
    generator = None if args.retrieval_only else Generator()

    rows = []
    for _, q in df.iterrows():
        res = answer_question(q["question"], retriever, generator, k=args.k, min_score=args.min_score)
        answerable = q["answerable"] == "yes"
        row = {
            "id": q["id"], "question": q["question"], "answerable": q["answerable"],
            "expected_answer": q["expected_answer"], "expected_page": q["expected_page"],
            "chatbot_answer": res["answer"], "abstained": res["abstained"],
            "top_score": round(res["top_score"], 3),
            "retrieved": "; ".join(f"{s['source']} p.{s['page']} ({s['score']:.2f})" for s in res["sources"]),
            "citation_hit": citation_hit(q["expected_page"], q["expected_source"], res["sources"]) if answerable else "n/a",
            "abstention_ok": "n/a" if answerable else ("yes" if res["abstained"] else "no"),
            "result": "", "notes": "",
        }
        rows.append(row)
        print(f"[{q['id']:>2}] hit={row['citation_hit']:<7} abstained={res['abstained']!s:<5} {q['question'][:70]}")

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(exist_ok=True)
    out.to_csv(out_path, index=False)

    ans, unans = out[out["answerable"] == "yes"], out[out["answerable"] == "no"]
    summary = {
        "n_questions": len(out), "n_answerable": len(ans), "n_unanswerable": len(unans),
        "top_k": args.k, "min_score": args.min_score, "retrieval_only": args.retrieval_only,
        "retrieval_all_expected_pages_found": int((ans["citation_hit"] == "yes").sum()),
        "retrieval_some_expected_pages_found": int((ans["citation_hit"] != "no").sum()),
        "correctly_abstained_on_unanswerable": int((unans["abstention_ok"] == "yes").sum()),
        "wrongly_abstained_on_answerable": int(ans["abstained"].sum()),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}. Now grade the `result` and `notes` columns by hand.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--min-score", type=float, default=MIN_SCORE)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())
