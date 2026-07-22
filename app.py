import os
import streamlit as st
from dotenv import load_dotenv
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq
import pytesseract
from pdf2image import convert_from_bytes
import re

load_dotenv()

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    POPPLER_PATH = r"C:\Windows\poppler-26.02.0\Library\bin"
else:
    POPPLER_PATH = None

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return embed_model, groq_client

embed_model, groq_client = load_models()

st.title("DocMind — Ask Your Document")
st.write("Upload a PDF (even scanned ones), then ask questions grounded in its content.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
        with st.spinner("Reading and OCR'ing your PDF... this may take a minute"):
            pdf_bytes = uploaded_file.read()
            pages = convert_from_bytes(pdf_bytes, poppler_path=POPPLER_PATH)

            full_text = ""
            for page_image in pages:
                full_text += pytesseract.image_to_string(page_image) + "\n"

            full_text = re.sub(r'\n{2,}', '\n', full_text)
            full_text = re.sub(r' {2,}', ' ', full_text)
            full_text = full_text.strip()

            # Chunking with LangChain's splitter (500 chars, 50 overlap)
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = [c for c in splitter.split_text(full_text) if len(c.strip()) > 20]

            # Embed chunks and build FAISS index (normalized -> inner product = cosine similarity)
            embeddings = embed_model.encode(chunks, normalize_embeddings=True)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(np.array(embeddings, dtype="float32"))

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.processed_file = uploaded_file.name

        st.success(f"Processed {len(chunks)} chunks from your document!")

    question = st.text_input("Your question:")

    if question:
        # Embed the question with the same model, retrieve top-3 chunks from FAISS
        q_emb = embed_model.encode([question], normalize_embeddings=True)
        scores, ids = st.session_state.index.search(np.array(q_emb, dtype="float32"), k=3)
        retrieved_chunks = [st.session_state.chunks[i] for i in ids[0]]
        context = "\n\n".join(retrieved_chunks)

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
                messages=[{"role": "user", "content": prompt}]
            )

        st.subheader("Answer")
        st.write(response.choices[0].message.content)

        with st.expander("Sources used"):
            for i, chunk in enumerate(retrieved_chunks):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(chunk)
