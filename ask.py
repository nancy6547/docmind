import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

client_db = chromadb.PersistentClient(path="./chroma_db")
collection = client_db.get_or_create_collection(name="docmind")

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

question = input("Ask something about your document: ")
question_embedding = embed_model.encode([question])

results = collection.query(
    query_embeddings=question_embedding.tolist(),
    n_results=3
)

retrieved_chunks = results['documents'][0]
context = "\n\n".join(retrieved_chunks)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

prompt = f"""Answer the question using ONLY the context below. 
If the answer isn't in the context, say "I don't know based on this document."

Context:
{context}

Question: {question}
"""

response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    max_tokens=300,
    messages=[{"role": "user", "content": prompt}]
)

print("\n--- Answer ---")
print(response.choices[0].message.content)

print("\n--- Sources used (raw) ---")
for i, c in enumerate(retrieved_chunks):
    print(f"\n--- Chunk {i+1} (length {len(c)}) ---")
    print(repr(c))