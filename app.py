"""DocMind: ask questions about any PDF, scanned ones included, and see which
sentences of the answer the document actually supports."""
import hashlib
import html
import json
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from docmind import core

load_dotenv()
NLI_MODEL = os.getenv("HALLUGUARD_MODEL", "cross-encoder/nli-deberta-v3-xsmall")
ROOT = Path(__file__).resolve().parent
SAMPLE_PDF = ROOT / "eval" / "handbook.pdf"
SAMPLE_QUESTIONS = ROOT / "eval" / "questions.json"
GREEN, AMBER, RED = "#1d8a4f", "#c27c0e", "#d62f2f"

st.set_page_config(page_title="DocMind · grounded PDF Q&A", page_icon="📄", layout="centered")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 860px;}
      .dm-sub {color: #6b7280; margin-top: -0.6rem; margin-bottom: 1rem;}
      .dm-legend span {display:inline-block; margin-right: 14px; font-size: 0.85rem;}
      .dm-dot {display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:middle;}
      .dm-answer {line-height: 1.9; font-size: 1.02rem;}
      .dm-chunk {border-left: 3px solid #e5e7eb; padding: 4px 0 4px 12px; margin: 6px 0 14px; color: #374151; font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- models: loaded once per server, shared by every visitor ---------------------------
@st.cache_resource(show_spinner="Loading the embedding model...")
def load_embedder():
    return core.sentence_transformer_embedder()


@st.cache_resource
def load_llm():
    return core.groq_llm()


@st.cache_resource(show_spinner="Loading the answer checker...")
def load_checker():
    from halluguard import NLIScorer

    return NLIScorer(NLI_MODEL)


# --- documents: keyed by content hash, so re-uploads and reruns never re-OCR -------------
@st.cache_resource(show_spinner=False, max_entries=16)
def process_pdf(digest: str, _pdf_bytes: bytes):
    ex = core.extract_text(_pdf_bytes)
    chunks = core.chunk(ex.text)
    return core.build_index(chunks, load_embedder()), (len(chunks), ex.text_pages, ex.ocr_pages)


def sample_questions() -> list[str]:
    try:
        data = json.loads(SAMPLE_QUESTIONS.read_text(encoding="utf-8"))
        return [x["q"] for x in data.get("answerable", [])][:3]
    except (OSError, ValueError, KeyError):
        return []


def highlight(answer_text: str, report) -> str:
    """Answer text with each sentence underlined green / amber / red by HalluGuard's verdict."""
    from halluguard import CONTRADICTED, SUPPORTED

    colors = {SUPPORTED: GREEN, CONTRADICTED: RED}
    parts, last = [], 0
    for c in report.claims:
        color = colors.get(c.verdict, AMBER)
        parts.append(html.escape(answer_text[last:c.start]))
        parts.append(
            f'<span title="{c.verdict} · support {c.support:.2f}" style="background:{color}1f;'
            f'border-bottom:2px solid {color}">{html.escape(c.text)}</span>'
        )
        last = c.end
    parts.append(html.escape(answer_text[last:]))
    return "".join(parts).replace("\n", "<br>")


def render_turn(turn: dict) -> None:
    """Draw one answered question (used for the live answer and for the history)."""
    if turn.get("html"):
        st.markdown(f'<div class="dm-answer">{turn["html"]}</div>', unsafe_allow_html=True)
    else:
        st.write(turn["text"])
    if turn.get("note"):
        kind, msg = turn["note"]
        (st.warning if kind == "warn" else st.caption)(msg)
    if turn.get("hits"):
        with st.expander(f"Sources used ({len(turn['hits'])})"):
            for i, (score, text) in enumerate(turn["hits"], start=1):
                st.markdown(f"**Chunk {i}** · similarity {score:.2f}")
                st.progress(min(max(score, 0.0), 1.0))
                st.markdown(f'<div class="dm-chunk">{html.escape(text)}</div>', unsafe_allow_html=True)
    if turn.get("timing"):
        st.caption(turn["timing"])


def ask(question: str, index, verify: bool, min_rel: float) -> dict:
    t0 = time.perf_counter()
    out = core.answer(question, index, load_llm(), min_relevance=min_rel)
    t1 = time.perf_counter()
    turn = {"text": out.text, "hits": out.hits, "html": None, "note": None}
    timing = f"Retrieval + answer {t1 - t0:.1f}s"
    if out.refused:
        turn["note"] = ("info", "The relevance guard stopped this before the LLM: nothing in the document is close enough to the question.")
    elif verify:
        from halluguard import SUPPORTED, check

        rep = check(out.text, out.context, load_checker(), threshold=0.5)
        timing += f" · checking {time.perf_counter() - t1:.1f}s"
        turn["html"] = highlight(out.text, rep)
        flagged = [c for c in rep.claims if c.verdict != SUPPORTED]
        if not rep.claims:
            turn["note"] = ("info", "Nothing to check in this answer.")
        elif flagged:
            turn["note"] = ("warn", f"{len(flagged)} of {len(rep.claims)} sentences aren't clearly supported by the document. Hover them for details.")
        else:
            turn["note"] = ("info", f"All {len(rep.claims)} sentences are supported by the retrieved text.")
    turn["timing"] = timing
    return turn


# --- page -----------------------------------------------------------------------------
st.title("📄 DocMind")
st.markdown(
    '<div class="dm-sub">Ask questions about any PDF, scanned ones included. Answers come only from the document, '
    'and every sentence is checked against it. '
    '<a href="https://github.com/nancy6547/docmind" target="_blank">Source on GitHub</a></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    verify = st.toggle("Check answers with HalluGuard", value=True, help="Each answer sentence is checked against the retrieved chunks with an NLI model.")
    min_rel = st.slider("Relevance guard", 0.0, 0.8, core.MIN_RELEVANCE, 0.05, help="If the best chunk is less similar than this, DocMind says it doesn't know instead of calling the LLM.")
    st.markdown(
        f'<div class="dm-legend"><span><i class="dm-dot" style="background:{GREEN}"></i>supported</span>'
        f'<span><i class="dm-dot" style="background:{AMBER}"></i>not in the source</span>'
        f'<span><i class="dm-dot" style="background:{RED}"></i>contradicted</span></div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("Pipeline: pypdf text layer → OCR only for scanned pages → hybrid search (embeddings + BM25) → relevance guard → LLM → HalluGuard sentence check.")

col1, col2 = st.columns([3, 1])
with col1:
    uploaded = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")
with col2:
    use_sample = st.button("Try a sample PDF", use_container_width=True, disabled=not SAMPLE_PDF.exists(),
                           help="A short fictional company handbook (one page is a scanned image, to show OCR), so you can try DocMind without uploading anything.")

if use_sample:
    st.session_state.doc = ("Sample: company handbook", SAMPLE_PDF.read_bytes())
elif uploaded is not None:
    st.session_state.doc = (uploaded.name, uploaded.getvalue())

doc = st.session_state.get("doc")
if doc is None:
    st.info("Upload a PDF, or click **Try a sample PDF** to see it work in a few seconds.")
    st.stop()

name, data = doc
digest = hashlib.sha256(data).hexdigest()
if st.session_state.get("digest") != digest:
    st.session_state.digest, st.session_state.history = digest, []

t0 = time.perf_counter()
with st.spinner("Reading your PDF (scanned pages are OCR'd, which can take a minute)..."):
    index, (n, tp, op) = process_pdf(digest, data)
st.success(f"**{name}**: {n} chunks · {tp} page(s) read directly · {op} page(s) via OCR · ready in {time.perf_counter() - t0:.1f}s")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        render_turn(turn)

suggested = None
if name.startswith("Sample") and not st.session_state.history:
    qs = sample_questions()
    if qs:
        st.caption("Try one of these:")
        cols = st.columns(len(qs))
        for c, q in zip(cols, qs):
            if c.button(q, use_container_width=True):
                suggested = q

question = st.chat_input("Ask a question about this document") or suggested
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Finding the answer and checking it..."):
            turn = ask(question, index, verify, min_rel)
        render_turn(turn)
    st.session_state.history.append({"q": question, **turn})
