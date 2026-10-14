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
b = res["guard"]["best"]
lines += ["", f"Relevance guard: the best threshold on this set is **{b['threshold']:.2f}**, which lets {b['answerable_let_through']:.0%} of answerable questions through and refuses {b['unanswerable_refused']:.0%} of unanswerable ones."]
if "answers" in res:
    a = res["answers"]
    lines.append(f"Answers: {a['accuracy']:.0%} contain the expected fact; {a['unanswerable_said_idk']:.0%} of unanswerable questions get \"I don't know\".")
    if "halluguard_fully_supported" in a:
        lines.append(f"HalluGuard: {a['halluguard_fully_supported']:.0%} of answers have every sentence supported by the retrieved text.")
readme = ROOT / "README.md"
t = readme.read_text(encoding="utf-8")
s, e = t.index("<!-- EVAL:START -->") + len("<!-- EVAL:START -->"), t.index("<!-- EVAL:END -->")
readme.write_text(t[:s] + "\n" + "\n".join(lines) + "\n" + t[e:], encoding="utf-8")
print("README updated")
sys.exit(0)
