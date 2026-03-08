
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import requests

st.title("⚡ EV AI Diagnostic Assistant")

st.write("Ask questions about EV repair issues")

# Download EV manual automatically
url = "https://github.com/streamlit/example-data/raw/master/uber-rides-data1.csv"

pdf_path = "manual.pdf"

try:
    r = requests.get(url)
    with open(pdf_path, "wb") as f:
        f.write(r.content)
except:
    st.error("Manual download failed")

# Load document
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector database
vectorstore = FAISS.from_documents(chunks, embeddings)

# Load LLM
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)

def generate_answer(context, question):

    prompt = f"""
You are an EV diagnostic assistant.

Manual:
{context}

Question:
{question}

Provide a clear diagnostic answer.
"""

    result = llm(prompt)

    return result[0]["generated_text"]


query = st.text_input("Ask your EV question")

if query:

    documents = vectorstore.similarity_search(query, k=3)

    context = " ".join([doc.page_content for doc in documents])

    sources = [doc.metadata["page"] for doc in documents]

    answer = generate_answer(context, query)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")

    for s in sources:
        st.write(f"Page: {s}")
