# Prototypes

The first command-line version of DocMind, kept for reference. It used **ChromaDB** with a persistent on-disk store:

- `ingest.py`: extract text with pypdf → chunk → embed → store in ChromaDB
- `ingest_ocr.py`: same, but with Tesseract OCR for scanned PDFs
- `ask.py`: query ChromaDB and answer with Groq
- `test.py`: first Groq API smoke test

The deployed app (`../app.py`) replaced ChromaDB with an in-memory FAISS index. These scripts need `pip install chromadb` to run.
