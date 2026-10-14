# DocMind

**Ask questions about any PDF, including scanned ones, and get answers grounded only in that document, with sources.**

**Live demo:** https://docmind-4trkdxlqhqz3mtx3wnmpja.streamlit.app/

DocMind is a retrieval-augmented generation (RAG) app built from scratch: text extraction with an OCR fallback, chunking, embeddings, a FAISS vector index, and an LLM that is only allowed to answer from the retrieved text.

---

## How it works

```
 PDF
  │
  ├─ page has a text layer? ──► pypdf extracts it directly        (fast)
  └─ scanned page?          ──► Poppler renders it → Tesseract OCR (only those pages)
  │
  ▼
 clean + chunk           RecursiveCharacterTextSplitter, 500 chars, 50 overlap
  │
  ▼
 embed                   all-MiniLM-L6-v2 (sentence-transformers), L2-normalized
  │
  ▼
 FAISS IndexFlatIP       inner product on normalized vectors = cosine similarity
  │
  ▼
 retrieve top-3  ──► best similarity < 0.25?  ──► "I don't know based on this document"
  │                                                (the LLM is never called)
  ▼
 Llama 3.3 70B (Groq)    prompt: answer ONLY from this context, otherwise say you don't know
  │
  ▼
 answer + the source chunks and their similarity scores
```

## Design decisions

- **OCR only where it's needed.** Each page's own text layer is tried first, and only pages with almost no text (under 50 characters) go through OCR. Digital PDFs process in seconds, and scanned ones still work.
- **Cosine similarity via FAISS inner product.** Embeddings are L2-normalized, so `IndexFlatIP` returns cosine similarity directly, and a flat (exact) index is plenty for a single document.
- **Two guards against hallucination.**
  1. *Retrieval guard:* if the best chunk's similarity is below 0.25, DocMind answers "I don't know" **without calling the LLM**.
  2. *Prompt guard:* the model is told to answer only from the given context, and to say it doesn't know otherwise.
- **Sources are always shown**, with similarity scores, so every answer can be checked against the document.

## What I learned building it

- **Debugging OCR.** The first version returned empty answers on scanned PDFs. Printing the extracted text length per page showed `0`: the PDF had no text layer at all. That led to the OCR pipeline, and later to the smarter OCR-only-where-needed version.
- **ChromaDB → FAISS.** The first prototype used ChromaDB with a persistent on-disk store (see [`prototypes/`](prototypes/)). For a per-upload, in-memory index, FAISS was simpler and faster, and it deployed cleanly on Streamlit Cloud.
- **Keeping secrets out of git.** An API key was once caught by GitHub push protection before it was pushed. It was rotated, and keys now live only in `.env` / Streamlit secrets.

## Tech stack

Python · Streamlit · pypdf · Tesseract + Poppler (OCR) · LangChain text splitters · sentence-transformers (`all-MiniLM-L6-v2`) · FAISS · Groq (Llama 3.3 70B)

## Run locally

```bash
git clone https://github.com/nancy6547/docmind.git
cd docmind
pip install -r requirements.txt
```

Install the OCR tools:

- **macOS:** `brew install tesseract poppler`
- **Ubuntu:** `sudo apt install tesseract-ocr poppler-utils`
- **Windows:** install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and [Poppler](https://github.com/oschwartz10612/poppler-windows/releases), then update the two paths at the top of `app.py`

Create a `.env` file containing `GROQ_API_KEY=your_key`, then:

```bash
streamlit run app.py
```

## Limitations

- One document at a time, and the index is rebuilt on each upload (it isn't persisted).
- Fixed-size character chunking can split a sentence or table across chunks.
- The 0.25 relevance threshold was chosen by hand, not tuned on an evaluation set.

## Roadmap

- [ ] Evaluation set (question → expected source page) to measure retrieval hit rate and tune the threshold
- [ ] Hybrid search (BM25 + embeddings) for exact names, numbers and codes
- [ ] Multi-document upload with per-document citations
- [ ] Answer verification with an NLI model (see [HalluGuard](https://github.com/nancy6547))

---

Built by **Nancy Singh**: [GitHub](https://github.com/nancy6547) · [LinkedIn](https://linkedin.com/in/nancy-singh-84367722a)
