import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

st.set_page_config(page_title="EV AI Diagnostic Assistant")

st.title("⚡ EV AI Diagnostic Assistant")
st.write("Ask questions about EV repair issues from the manual.")

# ---- Load PDF ----
pdf_path = "manual.pdf"

import requests

url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
pdf_path = "manual.pdf"

r = requests.get(url)

with open(pdf_path, "wb") as f:
    f.write(r.content)


loader = PyPDFLoader(pdf_path)
docs = loader.load()

# ---- Split text ----
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)

# ---- Embeddings ----
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---- Vector DB ----
vectorstore = FAISS.from_documents(chunks, embeddings)

# ---- LLM ----
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)

def generate_answer(context, question):

    prompt = f"""
You are an EV diagnostic assistant.

Use the EV repair manual below to answer the question.

Manual:
{context}

Question:
{question}

Provide a clear diagnostic explanation.
"""

    result = llm(prompt)

    return result[0]["generated_text"]


# ---- User input ----
query = st.text_input("Ask your EV question")

if query:

    documents = vectorstore.similarity_search(query, k=3)

    context = " ".join([doc.page_content for doc in documents])

    sources = [doc.metadata.get("page", "Unknown") for doc in documents]

    answer = generate_answer(context, query)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")

    for s in sources:
        st.write(f"Page: {s}")
