import os
import streamlit as st
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

# --- Load everything once, cached so it doesn't reload on every question ---
@st.cache_resource
def load_resources():
    client_db = chromadb.PersistentClient(path="./chroma_db")
    collection = client_db.get_or_create_collection(name="docmind")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return collection, embed_model, groq_client

collection, embed_model, groq_client = load_resources()

# --- UI ---
st.title("📄 DocMind — Ask Your Document")
st.write("Ask questions about your uploaded document. Answers are grounded only in its content.")

question = st.text_input("Your question:")

if question:
    question_embedding = embed_model.encode([question])
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=3
    )
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

    with st.expander("📚 Sources used"):
        for i, chunk in enumerate(retrieved_chunks):
            st.markdown(f"**Chunk {i+1}:**")
            st.text(chunk)