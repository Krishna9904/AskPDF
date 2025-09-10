# -----------------------------
# Imports
# -----------------------------
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from ollama import generate

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="RAG PDF QA",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# Functions
# -----------------------------
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_chunks(chunks, model):
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_chunks(question, chunks, index, embeddings, k=3):
    query_vec = model.encode([question])
    distances, indices = index.search(query_vec, k)
    retrieved = [chunks[i] for i in indices[0]]
    return retrieved

def ask_ollama(question, context_chunks):
    context_text = "\n".join(context_chunks)
    prompt = f"Context: {context_text}\nQuestion: {question}\nAnswer:"
    response = generate(model="llama3:latest", prompt=prompt)
    return response.response  # use .response instead of .text

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("RAG PDF QA Settings")
chunk_size = st.sidebar.number_input("Chunk size", min_value=100, max_value=2000, value=500, step=50)
overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=500, value=50, step=10)
num_retrievals = st.sidebar.number_input("Number of chunks to retrieve", min_value=1, max_value=10, value=3, step=1)

# -----------------------------
# Main UI
# -----------------------------
st.title("📄 RAG PDF QA with Ollama")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    st.info("Processing PDF...")
    pdf_text = load_pdf(uploaded_file)
    chunks = chunk_text(pdf_text, chunk_size=chunk_size, overlap=overlap)

    # Initialize embeddings model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_chunks(chunks, model)

    # Build FAISS index
    index = build_faiss_index(embeddings)

    st.success("PDF processed! You can now ask questions.")

    question = st.text_input("Ask a question about the PDF:")

    if question:
        with st.spinner("Generating answer..."):
            retrieved = retrieve_chunks(question, chunks, index, embeddings, k=num_retrievals)
            answer = ask_ollama(question, retrieved)
        st.markdown("**Answer:**")
        st.info(answer)
