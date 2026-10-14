import os
import re

import faiss
import numpy as np
import pytesseract
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_bytes
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    POPPLER_PATH = r"C:\Windows\poppler-26.02.0\Library\bin"
else:
    POPPLER_PATH = None

CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 50        # characters shared between neighbouring chunks
TOP_K = 3                 # chunks sent to the LLM
MIN_PAGE_CHARS = 50       # a page with less text than this is treated as scanned -> OCR
MIN_RELEVANCE = 0.25      # cosine similarity below this = "not in the document" (tunable)


@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return embed_model, groq_client


def clean(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def extract_text(pdf_bytes: bytes):
    """Use the PDF's own text layer when it has one; OCR only the pages that don't.

    Returns (full_text, text_pages, ocr_pages).
    """
    import io

    reader = PdfReader(io.BytesIO(pdf_bytes))
    page_texts = [(page.extract_text() or "") for page in reader.pages]
    needs_ocr = [i for i, t in enumerate(page_texts) if len(t.strip()) < MIN_PAGE_CHARS]

    if needs_ocr:
        images = convert_from_bytes(pdf_bytes, poppler_path=POPPLER_PATH)
        for i in needs_ocr:
            page_texts[i] = pytesseract.image_to_string(images[i])

    full_text = clean("\n".join(page_texts))
    return full_text, len(page_texts) - len(needs_ocr), len(needs_ocr)


embed_model, groq_client = load_models()

st.title("DocMind — Ask Your Document")
st.write("Upload a PDF (digital or scanned), then ask questions answered only from its content.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    if st.session_state.get("processed_file") != uploaded_file.name:
        with st.spinner("Reading your PDF (scanned pages are OCR'd, which can take a minute)..."):
            full_text, text_pages, ocr_pages = extract_text(uploaded_file.read())

            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks = [c for c in splitter.split_text(full_text) if len(c.strip()) > 20]

            # Normalized embeddings + inner product index = cosine similarity search
            embeddings = embed_model.encode(chunks, normalize_embeddings=True)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(np.array(embeddings, dtype="float32"))

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.processed_file = uploaded_file.name

        st.success(
            f"Processed {len(chunks)} chunks — {text_pages} page(s) read directly, {ocr_pages} page(s) via OCR."
        )

    question = st.text_input("Your question:")

    if question:
        q_emb = embed_model.encode([question], normalize_embeddings=True)
        scores, ids = st.session_state.index.search(np.array(q_emb, dtype="float32"), k=TOP_K)
        hits = [(float(s), st.session_state.chunks[i]) for s, i in zip(scores[0], ids[0]) if i != -1]

        st.subheader("Answer")

        # Relevance guard: if nothing in the document is close to the question, don't ask the LLM at all.
        if not hits or hits[0][0] < MIN_RELEVANCE:
            st.write("I don't know based on this document — nothing in it is closely related to that question.")
        else:
            context = "\n\n".join(chunk for _, chunk in hits)
            prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know based on this document."

Context:
{context}

Question: {question}
"""
            with st.spinner("Thinking..."):
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
            st.write(response.choices[0].message.content)

        with st.expander("Sources used"):
            for n, (score, chunk) in enumerate(hits, start=1):
                st.markdown(f"**Chunk {n}** · similarity {score:.2f}")
                st.write(chunk)
