"""Evaluate DocMind on a document with known answers.

    python eval/run_eval.py                      # retrieval + relevance guard (no API key needed)
    python eval/run_eval.py --llm                # + answer accuracy with Groq (needs GROQ_API_KEY)
    python eval/run_eval.py --llm --halluguard   # + check every answer sentence with HalluGuard

Measures:
  retrieval     is the evidence for each answerable question in the top-k chunks? (hit@1, hit@3, MRR)
  guard         the relevance guard should refuse unanswerable questions and let answerable ones through;
                swept over thresholds so MIN_RELEVANCE is picked from data, not guessed
  answers       (--llm) does the answer contain the expected fact? does it say "I don't know" when it should?
  grounding     (--halluguard) share of answers whose every sentence is supported by the retrieved chunks
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docmind import core  # noqa: E402


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def hashing_embedder(dim: int = 512):
    """Bag-of-words hashing embedder: no model download. For tests only."""
    import hashlib

    import numpy as np

    def embed(texts):
        out = np.zeros((len(texts), dim), dtype="float32")
        for r, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[r, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-9)

    return embed


def retrieval(qs, index, k):
    hit1 = hitk = rr = 0.0
    rows = []
    for item in qs:
        hits = index.search(item["q"], k)
        rank = next((i + 1 for i, (_, c) in enumerate(hits) if norm(item["evidence"]) in norm(c)), None)
        hit1 += rank == 1
        hitk += rank is not None
        rr += 1 / rank if rank else 0
        rows.append({"q": item["q"], "rank": rank, "top_score": hits[0][0] if hits else 0.0, "scanned": item.get("scanned", False)})
    n = len(qs)
    return {"hit@1": hit1 / n, f"hit@{k}": hitk / n, "mrr": rr / n, "rows": rows}


def guard_sweep(index, answerable, unanswerable):
    top = index.best_similarity
    a = [top(x["q"]) for x in answerable]
    u = [top(q) for q in unanswerable]
    rows = []
    for i in range(5, 70, 5):
        t = i / 100
        let_through = sum(s >= t for s in a) / len(a)
        refused = sum(s < t for s in u) / len(u)
        rows.append({"threshold": t, "answerable_let_through": let_through, "unanswerable_refused": refused, "balanced": (let_through + refused) / 2})
    best = max(rows, key=lambda r: (r["balanced"], r["answerable_let_through"]))
    return {"rows": rows, "best": best, "answerable_scores": a, "unanswerable_scores": u}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "eval/questions.json"))
    ap.add_argument("--k", type=int, default=core.TOP_K)
    ap.add_argument("--embedder", default="minilm", choices=["minilm", "hashing"])
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--halluguard", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "eval/results"))
    a = ap.parse_args()

    spec = json.loads(Path(a.questions).read_text(encoding="utf-8"))
    pdf = (ROOT / spec["document"]).read_bytes()
    t0 = time.time()
    ex = core.extract_text(pdf)
    chunks = core.chunk(ex.text)
    embed = core.sentence_transformer_embedder() if a.embedder == "minilm" else hashing_embedder()
    index = core.build_index(chunks, embed, mode="dense")
    print(f"{spec['document']}: {ex.text_pages} pages read directly, {ex.ocr_pages} OCR'd, {len(chunks)} chunks ({time.time() - t0:.1f}s)")

    res = {"embedder": a.embedder, "chunks": len(chunks), "ocr_pages": ex.ocr_pages}
    for mode in ("dense", "hybrid"):
        index.mode = mode
        rr = retrieval(spec["answerable"], index, a.k)
        scanned = [r for r in rr["rows"] if r["scanned"]]
        rr["scanned_hit"] = sum(r["rank"] is not None for r in scanned) / max(len(scanned), 1)
        res[f"retrieval_{mode}"] = rr
    index.mode = "hybrid"  # what the app uses
    res["retrieval"] = res["retrieval_hybrid"]
    res["guard"] = guard_sweep(index, spec["answerable"], spec["unanswerable"])

    r, g = res["retrieval"], res["guard"]
    print(f"\nRetrieval ({len(spec['answerable'])} questions, top {a.k} of {len(chunks)} chunks)")
    for mode in ("dense", "hybrid"):
        x = res[f"retrieval_{mode}"]
        print(f"  {mode:7s} hit@1 {x['hit@1']:.0%}  hit@{a.k} {x[f'hit@{a.k}']:.0%}  MRR {x['mrr']:.2f}  (scanned page: {x['scanned_hit']:.0%})")
    misses = [x["q"] for x in r["rows"] if x["rank"] is None]
    if misses:
        print("  hybrid missed:", *misses, sep="\n   - ")
    cur = min(g["rows"], key=lambda x: abs(x["threshold"] - core.MIN_RELEVANCE))
    print(f"\nRelevance guard at MIN_RELEVANCE={core.MIN_RELEVANCE}: lets {cur['answerable_let_through']:.0%} of answerable through, refuses {cur['unanswerable_refused']:.0%} of unanswerable")
    b = g["best"]
    print(f"Best threshold on this set: {b['threshold']:.2f} (lets {b['answerable_let_through']:.0%} through, refuses {b['unanswerable_refused']:.0%})")

    if a.llm:
        llm = core.groq_llm()
        checker = None
        if a.halluguard:
            from halluguard import NLIScorer

            scorer = NLIScorer()
            checker = lambda ans, ctx: __import__("halluguard").check(ans, ctx, scorer)  # noqa: E731
        correct = grounded = judged = 0
        for item in spec["answerable"]:
            out = core.answer(item["q"], index, llm)
            ok = not out.refused and norm(item["answer"]) in norm(out.text)
            correct += ok
            if checker and not out.refused:
                judged += 1
                grounded += not checker(out.text, out.context).hallucinated
        idk = 0
        for q in spec["unanswerable"]:
            out = core.answer(q, index, llm)
            idk += out.refused or "don't know" in out.text.lower()
        res["answers"] = {"accuracy": correct / len(spec["answerable"]), "unanswerable_said_idk": idk / len(spec["unanswerable"])}
        print(f"\nAnswers: {res['answers']['accuracy']:.0%} contain the right fact; {res['answers']['unanswerable_said_idk']:.0%} of unanswerable questions got \"I don't know\"")
        if checker:
            res["answers"]["halluguard_fully_supported"] = grounded / max(judged, 1)
            print(f"HalluGuard: {res['answers']['halluguard_fully_supported']:.0%} of answers have every sentence supported by the retrieved chunks")

    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, f"results_{a.embedder}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
