# PDF Semantic Search with FAISS

## 🔍 What this is

A simple semantic search tool for PDFs.  
It converts the document into embeddings and retrieves the most relevant text chunks for a given query using FAISS.

## ⚙️ Setup

Install required packages:

```bash
pip install numpy faiss-cpu sentence-transformers pypdf
```

## ▶️ Run

1. Place your PDF in the project folder
2. Update this line in the code:

```python
PDF_PATH = "your_document.pdf"
```

3. Run:

```bash
python main.py
```

4. Enter your query in the terminal

## 💬 Example queries

- What is the main topic of the document?
- Explain [any concept from the PDF]
