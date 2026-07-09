import pytesseract
from pdf2image import convert_from_path
import chromadb
from sentence_transformers import SentenceTransformer
import re

# --- Tell Python exactly where Tesseract and Poppler live ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Windows\poppler-26.02.0\Library\bin"

# --- Step A: Convert each PDF page into an image ---
print("Converting PDF pages to images...")
pages = convert_from_path("test.pdf", poppler_path=POPPLER_PATH)
print(f"Converted {len(pages)} pages.")

# --- Step B: Run OCR on each page image to extract text ---
full_text = ""
for i, page_image in enumerate(pages):
    print(f"Running OCR on page {i+1}/{len(pages)}...")
    text = pytesseract.image_to_string(page_image)
    full_text += text + "\n"

print(f"Total extracted characters: {len(full_text)}")
print("First 500 characters:")
print(repr(full_text[:500]))

# --- Step C: Clean the text ---
full_text = re.sub(r'\n{2,}', '\n', full_text)
full_text = re.sub(r' {2,}', ' ', full_text)
full_text = full_text.strip()

# --- Step D: Chunk the text ---
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
print(f"Split into {len(chunks)} usable chunks.")

# --- Step E: Embed and store ---
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="docmind")

collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print("Done! OCR'd chunks stored in vector database.")