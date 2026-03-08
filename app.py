import streamlit as st
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pypdf import PdfReader

st.title("⚡ EV AI Diagnostic Assistant")

st.write("Ask questions about EV repair issues")

# Download small test PDF
url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

pdf_path = "manual.pdf"

r = requests.get(url)

with open(pdf_path, "wb") as f:
    f.write(r.content)

# Read PDF
reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    text += page.extract_text()

# Simple chunking
chunk_size = 500
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

# LLM
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)

query = st.text_input("Ask your EV question")

if query:

    query_embedding = model.encode([query])

    distances, indices = index.search(np.array(query_embedding), k=3)

    context = ""

    for i in indices[0]:
        context += chunks[i] + "\n"

    prompt = f"""
You are an EV diagnostic assistant.

Manual:
{context}

Question:
{query}

Provide a clear diagnostic answer.
"""

    answer = llm(prompt)[0]["generated_text"]

    st.subheader("Answer")

    st.write(answer)
