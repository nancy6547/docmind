"""Pipeline tests with no model download: a hashing embedder and a fake LLM."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from docmind import core  # noqa: E402
from run_eval import hashing_embedder  # noqa: E402

TEXT = (
    "Employees get 24 days of paid annual leave per year. "
    "Salaries are paid on the last working day of each month. "
    "Robots in test mode are limited to 1.2 metres per second near people. "
) * 3


def test_chunks_are_bounded_and_nonempty():
    cs = core.chunk(TEXT, size=120, overlap=20)
    assert cs and all(len(c) <= 120 for c in cs)


def test_retrieval_finds_the_right_chunk():
    idx = core.build_index(core.chunk(TEXT, size=120, overlap=20), hashing_embedder())
    score, top = idx.search("how many days of annual leave", k=1)[0]
    assert "annual leave" in top and score > 0


def test_relevance_guard_skips_the_llm():
    idx = core.build_index(core.chunk(TEXT, size=120, overlap=20), hashing_embedder())
    called = []
    out = core.answer("zebra quantum saxophone", idx, lambda p: called.append(p) or "x", min_relevance=0.3)
    assert out.refused and not called


def test_answer_passes_context_to_llm():
    idx = core.build_index(core.chunk(TEXT, size=120, overlap=20), hashing_embedder())
    out = core.answer("annual leave days", idx, lambda p: "24 days" if "annual leave" in p else "?", min_relevance=0.0)
    assert not out.refused and out.text == "24 days" and "annual leave" in out.context


def test_ocr_only_where_needed():
    pdf = ROOT / "eval/handbook.pdf"
    ex = core.extract_text(pdf.read_bytes())
    assert ex.ocr_pages == 1 and ex.text_pages == 9
    assert "1.2 metres per second" in ex.text.replace("\n", " ")
