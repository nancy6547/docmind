"""Everything DocMind does, without Streamlit, so the app and the eval share one pipeline."""
from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field

CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50  # characters shared between neighbouring chunks
TOP_K = 3  # chunks sent to the LLM
MIN_PAGE_CHARS = 50  # a page with less text than this is treated as scanned -> OCR
MIN_RELEVANCE = 0.25  # best similarity below this = "not in the document" (see eval/README)
LLM_MODEL = "llama-3.3-70b-versatile"
EMBED_MODEL = "all-MiniLM-L6-v2"

if os.name == "nt":
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    POPPLER_PATH = os.getenv("POPPLER_PATH", r"C:\Windows\poppler-26.02.0\Library\bin")
else:
    TESSERACT_CMD, POPPLER_PATH = None, None


def clean(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


@dataclass
class Extracted:
    text: str
    text_pages: int  # pages read from the PDF's text layer
    ocr_pages: int  # pages that had no text layer and were OCR'd


def extract_text(pdf_bytes: bytes) -> Extracted:
    """Use the PDF's own text layer when it has one; OCR only the pages that don't."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [(p.extract_text() or "") for p in reader.pages]
    needs_ocr = [i for i, t in enumerate(pages) if len(t.strip()) < MIN_PAGE_CHARS]
    if needs_ocr:
        import pytesseract
        from pdf2image import convert_from_bytes

        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        for i in needs_ocr:  # convert only the pages that need it
            img = convert_from_bytes(pdf_bytes, first_page=i + 1, last_page=i + 1, poppler_path=POPPLER_PATH)[0]
            pages[i] = pytesseract.image_to_string(img)
    return Extracted(clean("\n".join(pages)), len(pages) - len(needs_ocr), len(needs_ocr))


def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return [c for c in splitter.split_text(text) if len(c.strip()) > 20]


_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Index:
    chunks: list[str]
    faiss_index: object
    embed: object  # callable: list[str] -> normalized float32 array
    bm25: object = None
    mode: str = "hybrid"  # "dense" or "hybrid"

    _qcache: dict = field(default_factory=dict, repr=False)

    def _qvec(self, question: str):
        """Embed each question once, even though search() and the relevance guard both need it."""
        import numpy as np

        if question not in self._qcache:
            if len(self._qcache) > 64:
                self._qcache.clear()
            self._qcache[question] = np.asarray(self.embed([question]), dtype="float32")
        return self._qcache[question]

    def dense(self, question: str, k: int) -> list[tuple[float, int]]:
        scores, ids = self.faiss_index.search(self._qvec(question), k)
        return [(float(s), int(i)) for s, i in zip(scores[0], ids[0]) if i != -1]

    def best_similarity(self, question: str) -> float:
        """Cosine similarity of the closest chunk: what the relevance guard checks."""
        d = self.dense(question, 1)
        return d[0][0] if d else 0.0

    def search(self, question: str, k: int = TOP_K) -> list[tuple[float, str]]:
        """Top-k chunks. The score returned is always the dense cosine similarity,
        so the relevance guard means the same thing in both modes."""
        n = len(self.chunks)
        dense = self.dense(question, n)
        if self.mode == "dense" or self.bm25 is None:
            return [(s, self.chunks[i]) for s, i in dense[:k]]
        # Hybrid: reciprocal rank fusion of embedding ranks and BM25 keyword ranks.
        # Embeddings catch paraphrases; BM25 catches exact numbers, names and codes.
        cos = {i: s for s, i in dense}
        bm = self.bm25.get_scores(tokenize(question))
        bm_rank = sorted(range(n), key=lambda i: -bm[i])
        fused: dict[int, float] = {}
        for r, (_, i) in enumerate(dense):
            fused[i] = fused.get(i, 0.0) + 1 / (60 + r + 1)
        for r, i in enumerate(bm_rank):
            if bm[i] > 0:
                fused[i] = fused.get(i, 0.0) + 1 / (60 + r + 1)
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [(cos.get(i, 0.0), self.chunks[i]) for i in top]


def sentence_transformer_embedder(model_name: str = EMBED_MODEL):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(list(texts), normalize_embeddings=True)


def build_index(chunks: list[str], embed, mode: str = "hybrid") -> Index:
    """Normalized embeddings + inner-product index = cosine similarity search, plus BM25."""
    import faiss
    import numpy as np
    from rank_bm25 import BM25Okapi

    emb = np.asarray(embed(chunks), dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return Index(chunks, index, embed, BM25Okapi([tokenize(c) or ["_"] for c in chunks]), mode)


PROMPT = """Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know based on this document."

Context:
{context}

Question: {question}
"""
REFUSAL = "I don't know based on this document — nothing in it is closely related to that question."


@dataclass
class Answer:
    text: str
    refused: bool  # True if the relevance guard stopped it before the LLM
    hits: list[tuple[float, str]] = field(default_factory=list)

    @property
    def context(self) -> str:
        return "\n\n".join(c for _, c in self.hits)


def answer(question: str, index: Index, llm, min_relevance: float = MIN_RELEVANCE, k: int = TOP_K) -> Answer:
    """Retrieve, apply the relevance guard, then ask the LLM. `llm(prompt) -> str`."""
    hits = index.search(question, k)
    best = index.best_similarity(question)
    if not hits or best < min_relevance:
        return Answer(REFUSAL, True, hits)
    ctx = "\n\n".join(c for _, c in hits)
    return Answer(llm(PROMPT.format(context=ctx, question=question)), False, hits)


def groq_llm(model: str = LLM_MODEL, api_key: str | None = None):
    from groq import Groq

    client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))

    def call(prompt: str) -> str:
        r = client.chat.completions.create(model=model, max_tokens=300, temperature=0, messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content

    return call
