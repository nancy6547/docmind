import os
import streamlit as st
from dotenv import load_dotenv
import chromadb
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

            def chunk_text(text, chunk_size=500, overlap=50):
                chunks = []
                start = 0
                while start < len(text):
                    end = start + chunk_size
                    chunk = text[start:end].strip()
                    if len(chunk) > 20:
                        chunks.append(chunk)
                    start += chunk_size - overlap
                return chunks

            chunks = chunk_text(full_text)
            embeddings = embed_model.encode(chunks)

            client_db = chromadb.Client()
            collection = client_db.get_or_create_collection(name="session_docs")
            collection.add(
                documents=chunks,
                embeddings=embeddings.tolist(),
                ids=[f"chunk_{i}" for i in range(len(chunks))]
            )

            st.session_state.collection = collection
            st.session_state.processed_file = uploaded_file.name

        st.success(f"Processed {len(chunks)} chunks from your document!")

    question = st.text_input("Your question:")

    if question:
        collection = st.session_state.collection
        question_embedding = embed_model.encode([question])
        results = collection.query(query_embeddings=question_embedding.tolist(), n_results=3)
        retrieved_chunks = results['documents'][0]
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
                st.text(chunk)