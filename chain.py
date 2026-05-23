# chain.py

# ── Imports ───────────────────────────────────────────────────────────────────
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import fitz  # ← NEW: pymupdf, better PDF reader than PyPDFLoader

load_dotenv()


# ── NEW: Better PDF loader ────────────────────────────────────────────────────
# PyPDFLoader mixes columns together in academic papers
# fitz reads each page properly and preserves structure
def load_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n\n"
    doc.close()
    return text


# ── Load embedding model ONCE at startup ──────────────────────────────────────
# This runs when chain.py is first imported, not on every PDF upload
# Saves 8-10 seconds per upload
print("Loading embedding model...")
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model ready.")


def build_chain(file_path):

    # ── Step 1: Load PDF ──────────────────────────────────────────────────────
    raw_text = load_pdf(file_path)

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    docs = splitter.create_documents([raw_text])

    # ── Step 3: Embed and store ───────────────────────────────────────────────
    # CHANGED: uses the pre-loaded EMBEDDINGS instead of creating a new one
    vectorstore = FAISS.from_documents(docs, EMBEDDINGS)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 8})

    # ── Step 4: LLM ───────────────────────────────────────────────────────────
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
    )

    # ── Step 5: Prompt template ───────────────────────────────────────────────
    # FIXED: removed the stray "s" at the end of the prompt
    prompt = ChatPromptTemplate.from_template("""
You are a helpful research assistant analyzing an academic paper.
Use ONLY the context below to answer. If the context does not contain
enough information, say "The retrieved sections don't contain this information."

Context:
{context}

Question:
{question}

Answer:
""")

    # ── Helper function ───────────────────────────────────────────────────────
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # ── Step 6: Build the chain ───────────────────────────────────────────────
    chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    chain._vectorstore = vectorstore
    return chain