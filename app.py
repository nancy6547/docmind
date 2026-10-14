"""DocMind: ask questions about any PDF, scanned ones included, and see which
sentences of the answer the document actually supports."""
import html
import os

import streamlit as st
from dotenv import load_dotenv

from docmind import core

load_dotenv()
NLI_MODEL = os.getenv("HALLUGUARD_MODEL", "cross-encoder/nli-deberta-v3-xsmall")

st.set_page_config(page_title="DocMind", page_icon="📄")


@st.cache_resource
def load_embedder():
    return core.sentence_transformer_embedder()


@st.cache_resource
def load_llm():
    return core.groq_llm()


@st.cache_resource
def load_checker():
    from halluguard import NLIScorer

    return NLIScorer(NLI_MODEL)


def highlight(answer_text: str, report) -> str:
    """Answer text with each sentence underlined green / amber / red by HalluGuard's verdict."""
    from halluguard import CONTRADICTED, SUPPORTED

    colors = {SUPPORTED: "#1d8a4f", CONTRADICTED: "#d62f2f"}
    parts, last = [], 0
    for c in report.claims:
        color = colors.get(c.verdict, "#c27c0e")
        parts.append(html.escape(answer_text[last:c.start]))
        parts.append(
            f'<span title="{c.verdict} · support {c.support:.2f}" style="background:{color}1f;'
            f'border-bottom:2px solid {color}">{html.escape(c.text)}</span>'
        )
        last = c.end
    parts.append(html.escape(answer_text[last:]))
    return "".join(parts).replace("\n", "<br>")


st.title("DocMind")
st.write("Upload a PDF (digital or scanned) and ask questions. Answers come only from the document, and every sentence is checked against it.")

with st.sidebar:
    st.header("Settings")
    verify = st.toggle("Check answers with HalluGuard", value=True, help="Each answer sentence is checked against the retrieved chunks with an NLI model.")
    min_rel = st.slider("Relevance guard", 0.0, 0.8, core.MIN_RELEVANCE, 0.05, help="If the best chunk is less similar than this, DocMind says it doesn't know instead of calling the LLM.")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded is not None:
    if st.session_state.get("processed_file") != uploaded.name:
        with st.spinner("Reading your PDF (scanned pages are OCR'd, which can take a minute)..."):
            ex = core.extract_text(uploaded.read())
            chunks = core.chunk(ex.text)
            st.session_state.index = core.build_index(chunks, load_embedder())
            st.session_state.processed_file = uploaded.name
            st.session_state.stats = (len(chunks), ex.text_pages, ex.ocr_pages)
    n, tp, op = st.session_state.stats
    st.success(f"Processed {n} chunks: {tp} page(s) read directly, {op} page(s) via OCR.")

    question = st.text_input("Your question:")
    if question:
        with st.spinner("Thinking..."):
            out = core.answer(question, st.session_state.index, load_llm(), min_relevance=min_rel)

        st.subheader("Answer")
        if out.refused or not verify:
            st.write(out.text)
        else:
            from halluguard import SUPPORTED, check

            with st.spinner("Checking every sentence against the document..."):
                rep = check(out.text, out.context, load_checker(), threshold=0.5)
            st.markdown(f'<div style="line-height:1.9">{highlight(out.text, rep)}</div>', unsafe_allow_html=True)
            flagged = [c for c in rep.claims if c.verdict != SUPPORTED]
            if not rep.claims:
                st.caption("Nothing to check in this answer.")
            elif flagged:
                st.warning(f"{len(flagged)} of {len(rep.claims)} sentences aren't clearly supported by the document. Hover them for details.")
            else:
                st.caption(f"All {len(rep.claims)} sentences are supported by the retrieved text.")

        with st.expander("Sources used"):
            for i, (score, text) in enumerate(out.hits, start=1):
                st.markdown(f"**Chunk {i}** · similarity {score:.2f}")
                st.write(text)
