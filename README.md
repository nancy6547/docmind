# DocMind

[![test](https://github.com/nancy6547/docmind/actions/workflows/test.yml/badge.svg)](https://github.com/nancy6547/docmind/actions/workflows/test.yml)

**Ask questions about any PDF, including scanned ones. Answers come only from the document, and every sentence of the answer is checked against it.**

**Live demo:** https://docmind-4trkdxlqhqz3mtx3wnmpja.streamlit.app/

DocMind is a retrieval-augmented generation (RAG) app built from scratch. It has text extraction with an OCR fallback, hybrid search (embeddings + BM25), a relevance guard, and an LLM that may only answer from the retrieved text. Each answer sentence is then verified with [HalluGuard](https://github.com/nancy6547/halluguard) and highlighted green, amber or red. It ships with an evaluation set, so retrieval quality and the guard's threshold are measured, not guessed.

---

## How it works

```
 PDF
  ├─ page has a text layer? ──► pypdf extracts it directly          (fast)
  └─ scanned page?          ──► Poppler renders that page → Tesseract OCR
  ▼
 clean + chunk           500 chars, 50 overlap
  ▼
 index                   all-MiniLM-L6-v2 embeddings in FAISS (cosine)  +  BM25 keyword index
  ▼
 question ──► closest chunk's cosine < 0.25? ──► "I don't know based on this document"
  │                                              (the LLM is never called)
  ▼
 hybrid retrieve top-3   reciprocal rank fusion of embedding ranks and BM25 ranks
  ▼
 Llama 3.3 70B (Groq)    answer ONLY from this context, otherwise say you don't know
  ▼
 HalluGuard              every answer sentence checked against the retrieved chunks with an NLI model
  ▼
 answer with each sentence marked supported / not in the document / contradicted, plus the source chunks
```

## Design decisions

- **OCR only where it's needed.** Each page's text layer is tried first, and only pages with under 50 characters of text are rendered and OCR'd, one page at a time. Digital PDFs process in seconds, and scanned ones still work.
- **Hybrid retrieval.** Embeddings find paraphrases ("work from home" ≈ "remote"). BM25 finds exact tokens embeddings blur: numbers, names, codes. The two rankings are merged with reciprocal rank fusion, so neither score scale has to be tuned against the other.
- **Three guards against hallucination.**
  1. *Relevance guard:* if no chunk is close to the question, DocMind says "I don't know" **without calling the LLM**.
  2. *Prompt guard:* the model must answer only from the given context.
  3. *Answer check:* HalluGuard runs an NLI cross-encoder over every answer sentence and flags the ones the retrieved text doesn't support.
- **Measured, not guessed.** `eval/` contains a 10-page test handbook (one page is a scan with no text layer) with 25 answerable and 10 deliberately unanswerable questions.
- **One pipeline.** `docmind/core.py` holds all the logic, without Streamlit. The app and the evaluation call the same code, and the tests run without downloading a model.

## Evaluation

```bash
python eval/run_eval.py                      # retrieval + relevance guard, no API key needed
python eval/run_eval.py --llm --halluguard   # + answer accuracy and grounding (needs GROQ_API_KEY)
```

It reports:

- **Retrieval:** is the evidence sentence in the top 3 chunks (hit@1, hit@3, MRR), dense vs. hybrid, including the questions whose answer sits on the scanned page.
- **Relevance guard:** for every threshold from 0.05 to 0.65, the share of answerable questions let through and of unanswerable ones refused.
- **Answers** (`--llm`): does the answer contain the expected fact, and do unanswerable questions get "I don't know"?
- **Grounding** (`--halluguard`): share of answers with every sentence supported.

<!-- EVAL:START -->
Results on `eval/handbook.pdf` (12 chunks, 1 scanned page), all-MiniLM-L6-v2, top 3:

| Retrieval | hit@1 | hit@3 | MRR | Scanned-page questions found |
|---|---|---|---|---|
| dense | 72% | 88% | 0.80 | 50% |
| hybrid | 80% | 96% | 0.88 | 83% |

Hybrid search puts the right chunk in the top 3 for **96%** of questions, vs 88% with embeddings alone. The biggest gain is on the scanned page (50% → 83%), likely because exact keywords still match when OCR text is messy.

Relevance guard: at the current setting (0.25) it lets 100% of answerable questions through but refuses only 20% of unanswerable ones on its own; the LLM's "I don't know" instruction is the second line of defence. On this set 0.35 balances best (88% through, 60% refused), but that's picked on the same questions it's scored on, so the default stays until it's checked on a separate question set.
<!-- EVAL:END -->

## What I learned building it

- **Debugging OCR.** The first version returned empty answers on scanned PDFs. Printing the extracted text length per page showed `0`: the PDF had no text layer at all. That led to the OCR pipeline, and later to the OCR-only-where-needed version.
- **ChromaDB → FAISS.** The first prototype used ChromaDB with a persistent on-disk store (see [`prototypes/`](prototypes/)). For a per-upload, in-memory index, FAISS was simpler and faster, and it deployed cleanly on Streamlit Cloud.
- **Keeping secrets out of git.** An API key was once caught by GitHub push protection before it was pushed. It was rotated, and keys now live only in `.env` / Streamlit secrets.

## Tech stack

Python · Streamlit · pypdf · Tesseract + Poppler (OCR) · LangChain text splitters · sentence-transformers (`all-MiniLM-L6-v2`) · FAISS · rank-bm25 · Groq (Llama 3.3 70B) · HalluGuard (NLI, DeBERTa-v3) · pytest

## Run locally

```bash
git clone https://github.com/nancy6547/docmind.git
cd docmind
pip install -r requirements.txt
```

Install the OCR tools:

- **macOS:** `brew install tesseract poppler`
- **Ubuntu:** `sudo apt install tesseract-ocr poppler-utils`
- **Windows:** install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and [Poppler](https://github.com/oschwartz10612/poppler-windows/releases). If they're not in the default locations, set `TESSERACT_CMD` and `POPPLER_PATH` in `.env`.

Create a `.env` file containing `GROQ_API_KEY=your_key`, then:

```bash
streamlit run app.py
pytest -q          # tests: no API key or model download needed
```

## Limitations

- One document at a time, and the index is rebuilt on each upload (it isn't persisted).
- Fixed-size character chunking can split a sentence or table across chunks.
- The evaluation document is small (10 pages, 35 questions); it catches regressions but isn't a broad benchmark. HalluGuard has its own benchmark on RAGTruth.
- HalluGuard checks against the *retrieved* chunks, so a fact that is in the document but wasn't retrieved is marked as unsupported.

## Roadmap

- [x] Evaluation set to measure retrieval and tune the relevance threshold
- [x] Hybrid search (BM25 + embeddings) for exact names, numbers and codes
- [x] Answer verification with an NLI model ([HalluGuard](https://github.com/nancy6547/halluguard))
- [ ] Multi-document upload with per-document citations
- [ ] Page numbers in citations

---

Built by **Nancy Singh**: [GitHub](https://github.com/nancy6547) · [LinkedIn](https://linkedin.com/in/nancy-singh-84367722a) · [Portfolio](https://nancy6547.github.io)
