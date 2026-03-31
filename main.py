import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import textwrap
import os

# --- 1. Configuration ---
PDF_PATH = "sample.pdf"  # <-- CHANGE THIS TO YOUR PDF FILE PATH
WORDS_PER_CHUNK = 150
TOP_K = 5

print("Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")


# --- 2. PDF Extraction & Chunking ---
def extract_text_from_pdf(pdf_path):
    """Reads a PDF and extracts all text into a single string."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Could not find PDF at: {pdf_path}")

    reader = PdfReader(pdf_path)
    full_text = ""

    print(f"Extracting text from {len(reader.pages)} pages...")
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            full_text += extracted + " "

    # Clean up excess whitespace and newlines
    return " ".join(full_text.split())


def chunk_text(text, chunk_size=WORDS_PER_CHUNK):
    """Splits text into chunks of exactly `chunk_size` words."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

    return chunks


# --- 3. Build the FAISS Index ---
def build_faiss_index(chunks):
    """Embeds chunks and loads them into a FAISS Inner Product index."""
    print(
        f"Generating embeddings for {len(chunks)} chunks... (This may take a moment for large PDFs)"
    )
    embeddings = model.encode(chunks)

    # Normalize for Cosine Similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


# --- 4. Main Execution & CLI ---
def main():
    # Setup data
    try:
        raw_text = extract_text_from_pdf(PDF_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    chunks = chunk_text(raw_text, WORDS_PER_CHUNK)
    print(f"Document split into {len(chunks)} chunks of ~{WORDS_PER_CHUNK} words.")

    index = build_faiss_index(chunks)
    print("FAISS Index successfully built!\n")

    # CLI Loop
    print("=" * 60)
    print(" PDF RAG RETRIEVER CLI (Type 'exit' to quit)")
    print("=" * 60)

    while True:
        query = input("\nQuery: ").strip()

        if query.lower() in ["exit", "quit"]:
            break
        if not query:
            continue

        # Embed and normalize the query
        query_vector = model.encode([query])
        faiss.normalize_L2(query_vector)

        # Search the index
        distances, indices = index.search(query_vector, TOP_K)

        print("\nTop 5 relevant chunks:\n")
        for rank, (score, idx) in enumerate(zip(distances[0], indices[0])):
            retrieved_text = chunks[idx]
            wrapped_text = textwrap.fill(retrieved_text, width=80)

            print(f"[Result {rank + 1} | Similarity: {score:.4f}]")
            print(f"{wrapped_text}")
            print("-" * 40)


if __name__ == "__main__":
    main()
