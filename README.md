# DocMind 

An AI-powered document assistant that enables users to chat with PDFs using Retrieval-Augmented Generation (RAG).

## Features

- 📄 Upload PDF documents
- 🔍 OCR support for scanned PDFs
- 🧩 Intelligent text chunking
- 🧠 Semantic search using embeddings
- 🗂️ Vector database retrieval
- 🤖 Context-aware answers using an LLM
- 🌐 Streamlit web interface

## Tech Stack

- Python
- Streamlit
- Sentence Transformers
- Vector Database (mention the one you used)
- OCR library (mention which one)
- LLM (mention which one)
- LangChain (only if you actually used it)

## Architecture

PDF
↓
OCR (if required)
↓
Chunking
↓
Embeddings
↓
Vector Database
↓
Retriever
↓
LLM
↓
Answer

## How It Works

1. Upload a PDF.
2. Extract text (OCR if needed).
3. Split text into chunks.
4. Generate embeddings.
5. Store embeddings in the vector database.
6. Retrieve the most relevant chunks.
7. Generate a grounded response using the LLM.

## Future Improvements

- Hybrid Search
- Agentic RAG
- Better evaluation
- Multi-document support
- Codebase support

## Demo

(https://docmind-4trkdxlqhqz3mtx3wnmpja.streamlit.app/)

## Lessons Learned

This project helped me understand:
- RAG architecture
- Embeddings
- Semantic search
- Vector databases
- Prompt construction
- Deployment
