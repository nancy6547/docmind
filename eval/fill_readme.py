"""Write the latest eval results into README.md between the EVAL markers."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
res = json.loads((ROOT / "eval/results/results_minilm.json").read_text(encoding="utf-8"))
lines = [
    f"Results on `eval/handbook.pdf` ({res['chunks']} chunks, {res['ocr_pages']} scanned page), all-MiniLM-L6-v2, top 3:",
    "",
    "| Retrieval | hit@1 | hit@3 | MRR | Scanned-page questions found |",
    "|---|---|---|---|---|",
]
for m in ("dense", "hybrid"):
    r = res[f"retrieval_{m}"]
    lines.append(f"| {m} | {r['hit@1']:.0%} | {r['hit@3']:.0%} | {r['mrr']:.2f} | {r['scanned_hit']:.0%} |")
d, h = res["retrieval_dense"], res["retrieval_hybrid"]
lines += ["", f"Hybrid search puts the right chunk in the top 3 for **{h['hit@3']:.0%}** of questions, vs {d['hit@3']:.0%} with embeddings alone. "
          f"The biggest gain is on the scanned page ({d['scanned_hit']:.0%} \u2192 {h['scanned_hit']:.0%}), likely because exact keywords still match when OCR text is messy."]
sys.path.insert(0, str(ROOT))
MIN = 0.25  # docmind.core.MIN_RELEVANCE (kept here so this script needs no ML packages)
cur = min(res["guard"]["rows"], key=lambda r: abs(r["threshold"] - MIN))
b = res["guard"]["best"]
lines += ["", f"Relevance guard: at the current setting ({MIN:.2f}) it lets {cur['answerable_let_through']:.0%} of answerable questions through "
          f"but refuses only {cur['unanswerable_refused']:.0%} of unanswerable ones on its own; the LLM's \"I don't know\" instruction is the second line of defence. "
          f"On this set {b['threshold']:.2f} balances best ({b['answerable_let_through']:.0%} through, {b['unanswerable_refused']:.0%} refused), "
          "but that's picked on the same questions it's scored on, so the default stays until it's checked on a separate question set."]
if "answers" in res:
    a = res["answers"]
    lines.append(f"Answers: {a['accuracy']:.0%} contain the expected fact; {a['unanswerable_said_idk']:.0%} of unanswerable questions get \"I don't know\".")
    if "halluguard_fully_supported" in a:
        lines.append(f"HalluGuard: {a['halluguard_fully_supported']:.0%} of answers have every sentence supported by the retrieved text.")
readme = ROOT / "README.md"
t = readme.read_text(encoding="utf-8").replace("\r\n", "\n")
s, e = t.index("<!-- EVAL:START -->") + len("<!-- EVAL:START -->"), t.index("<!-- EVAL:END -->")
readme.write_text(t[:s] + "\n" + "\n".join(lines) + "\n" + t[e:], encoding="utf-8", newline="\n")
print("README updated")
sys.exit(0)
