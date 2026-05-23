# chain.py

# ── Imports ───────────────────────────────────────────────────────────────────
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# This reads your .env file and loads GROQ_API_KEY into the environment
# so ChatGroq can find it automatically
load_dotenv()


def build_chain(file_path):

    # ── Step 1: Load PDF ──────────────────────────────────────────────────────
    # PyPDFLoader reads the PDF and gives us a list of Document objects
    # Each Document has page_content (the text) and metadata (page number etc)
    loader = PyPDFLoader(file_path)
    pages  = loader.load()

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    # We can't send the whole document to the LLM at once
    # So we split it into smaller pieces (chunks)
    # chunk_size=1000 → each chunk is max 1000 characters
    # chunk_overlap=100 → consecutive chunks share 100 characters
    # so sentences don't get cut off at the boundary between chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = splitter.split_documents(pages)

    # ── Step 3: Embed and store ───────────────────────────────────────────────
    # Convert each chunk into a vector (list of numbers)
    # Similar chunks will have similar vectors
    # all-MiniLM-L6-v2 is a small, fast embedding model (~90MB)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # FAISS stores all the vectors in memory
    # so we can search them quickly when a question comes in
    vectorstore = FAISS.from_documents(docs, embeddings)

    # as_retriever() wraps the vectorstore so it can be used in a chain
    # k=2 means return the 2 most similar chunks for each question
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # ── Step 4: LLM ───────────────────────────────────────────────────────────
    # ChatGroq connects to Groq's API
    # model: llama-3.1-8b-instant is fast and free
    # temperature=0 means deterministic answers (no randomness)
    # higher temperature = more creative but less consistent
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
    )

    # ── Step 5: Prompt template ───────────────────────────────────────────────
    # This is what gets sent to the LLM for every question
    # {context} gets replaced with the retrieved chunks
    # {question} gets replaced with the user's question
    # "Answer only from context" prevents hallucination
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions about a document.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't know based on the provided document."
Do not make up information.

Context:
{context}

Question:
{question}

Answer:
""")

    # ── Helper function ───────────────────────────────────────────────────────
    # The retriever returns a list of Document objects
    # The prompt expects a single string for {context}
    # This function joins all chunks into one string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # ── Step 6: Build the chain ───────────────────────────────────────────────
    # This is LCEL — read it like a pipeline left to right:
    #
    # 1. Input question comes in
    # 2. retriever finds the 2 most relevant chunks
    # 3. format_docs joins them into a string → fills {context}
    # 4. RunnablePassthrough passes the question through unchanged → fills {question}
    # 5. prompt builds the full text to send to the LLM
    # 6. llm generates the answer
    # 7. StrOutputParser extracts the plain string from the LLM response
    chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # We store the vectorstore on the chain object
    # so we can access it later for tracing (Layer 1)
    chain._vectorstore = vectorstore

    return chain