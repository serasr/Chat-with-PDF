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
from tracer import Trace, classify_error, ErrorType

# ── Constants ─────────────────────────────────────────────────────────────────
RETRIEVAL_SCORE_THRESHOLD = 1.8  # FAISS L2 distance - above this = poor match
MIN_CHUNKS_REQUIRED       = 1    # if we get 0 chunks, don't call the LLM

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

    # ── Check: did we get any text? ───────────────────────────────────────────
    if not raw_text or len(raw_text.strip()) < 100:
        raise ValueError(
            "empty_document: PDF has no extractable text. "
            "It may be a scanned image. Try a text-based PDF."
        )

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
Use the context below to answer the question as best you can.
If the context is insufficient, say what you do know and note the limitation.

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


def traced_query(chain, query: str) -> str:
    """
    Run the chain with full tracing and error classification.

    Error handling flow:
    1. Check if retrieval found anything useful
    2. If retrieval failed → return early, don't waste an LLM call
    3. If LLM fails → classify the error, log it, return friendly message
    4. All errors go to the tracer with classification
    """
    trace = Trace(query)

    try:
        # ── Span 1: retrieval ─────────────────────────────────────────────
        with trace.span("retrieval") as r_span:
            docs_with_scores = chain._vectorstore.similarity_search_with_score(
                query, k=8
            )
            chunks_retrieved = len(docs_with_scores)
            top_score = round(float(docs_with_scores[0][1]), 4) if docs_with_scores else None

            r_span.metadata["chunks_retrieved"] = chunks_retrieved
            r_span.metadata["top_score"]        = top_score

        # ── Check 1: did retrieval find anything? ─────────────────────────
        if chunks_retrieved < MIN_CHUNKS_REQUIRED:
            msg = "I couldn't find any relevant sections in the document for your question."
            trace.finish(
                answer     = msg,
                error      = "retrieval returned 0 chunks",
                error_type = ErrorType.RETRIEVAL_FAILURE
            )
            return msg

        # ── Check 2: are the chunks relevant enough? ──────────────────────────
        if top_score and top_score > RETRIEVAL_SCORE_THRESHOLD:
            msg = "Your question doesn't seem related to the document. Please ask something about the uploaded PDF."
            print(f"  [WARN] Low retrieval score: {top_score} — blocking LLM call")
            trace.finish(
                answer     = msg,
                error      = f"low retrieval score: {top_score}",
                error_type = ErrorType.RETRIEVAL_FAILURE
            )
            return msg

        # ── Span 2: LLM call ──────────────────────────────────────────────
        with trace.span("llm") as llm_span:
            answer = chain.invoke(query)

            # ── Check 3: did the LLM return anything? ─────────────────────
            if not answer or len(answer.strip()) == 0:
                raise ValueError("LLM returned empty response")

            llm_span.metadata["token_estimate"] = len(answer) // 4

        trace.finish(answer=answer)
        return answer

    except ValueError as e:
        error_type = ErrorType.LLM_FAILURE
        msg        = "The model returned an empty response. Please try again."
        trace.finish(error=str(e), error_type=error_type)
        return msg

    except Exception as e:
        error_type = classify_error(e)
        print(f"\n  [ERROR] {error_type}: {str(e)}")

        # Friendly messages per error type
        messages = {
            ErrorType.API_ERROR:    "The AI service is temporarily unavailable. Please try again in a moment.",
            ErrorType.LLM_FAILURE:  "The model took too long to respond. Please try again.",
            ErrorType.EMPTY_DOCUMENT: "The document appears to have no readable text.",
            ErrorType.UNKNOWN:      "Something went wrong. Please try again.",
        }

        msg = messages.get(error_type, messages[ErrorType.UNKNOWN])
        trace.finish(error=str(e), error_type=error_type)
        return msg