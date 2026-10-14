from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

# --- Step A: Extract text from the PDF ---
reader = PdfReader("test.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

print(f"Extracted {len(full_text)} characters from the PDF.")



print(f"Extracted {len(full_text)} characters from the PDF.")
print("First 500 characters of extracted text:")
print(repr(full_text[:500]))

# --- Step B: Chunk the text ---
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(full_text)
print(f"Split into {len(chunks)} chunks.")

# --- Step C: Load embedding model ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Step D: Create embeddings for each chunk ---
embeddings = model.encode(chunks)

# --- Step E: Store in ChromaDB ---
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="docmind")

collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print("Done! Chunks stored in vector database.")