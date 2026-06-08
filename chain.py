# ── Imports ───────────────────────────────────────────────────────────────────
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from dotenv import load_dotenv
from typing import List
from pydantic import Field
import fitz
from rank_bm25 import BM25Okapi
from tracer import Trace, classify_error, ErrorType

# ── Constants ─────────────────────────────────────────────────────────────────
RETRIEVAL_SCORE_THRESHOLD = 1.8
MIN_CHUNKS_REQUIRED       = 1
TOP_K                     = 8   # how many chunks to retrieve total

load_dotenv()


# ── PDF loader ────────────────────────────────────────────────────────────────
def load_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n\n"
    doc.close()
    return text


# ── Embedding model - loaded once at startup ──────────────────────────────────
print("Loading embedding model...")
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model ready.")


# ── RRF - Reciprocal Rank Fusion ──────────────────────────────────────────────
def reciprocal_rank_fusion(results_list: list, k: int = 60) -> list:
    """
    Combine multiple ranked result lists into one ranked list.

    How it works:
    - Each chunk gets a score for each list it appears in: 1/(k + rank)
    - Scores are summed across all lists
    - Chunks appearing in multiple lists get higher combined scores
    - Final list is sorted by combined score

    k=60 is the standard constant that smooths out rank differences.
    """
    scores = {}

    for results in results_list:
        for rank, chunk in enumerate(results):
            # use page_content as unique identifier for each chunk
            chunk_id = chunk.page_content
            if chunk_id not in scores:
                scores[chunk_id] = {"score": 0.0, "chunk": chunk}
            scores[chunk_id]["score"] += 1.0 / (k + rank + 1)

    # sort by combined score descending
    sorted_chunks = sorted(
        scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return [item["chunk"] for item in sorted_chunks]


# ── Hybrid Retriever ──────────────────────────────────────────────────────────
class HybridRetriever(BaseRetriever):
    """
    Combines BM25 (keyword) and FAISS (semantic) search using RRF.

    BM25  → finds chunks with exact keyword matches
    FAISS → finds chunks with similar meaning
    RRF   → combines both ranked lists into one

    This fixes the core problem: semantic search alone misses
    chunks that contain exact query terms like "abstract" or "BERTScore"
    """

    vectorstore: object = Field(description="FAISS vectorstore")
    bm25: object        = Field(description="BM25 index")
    docs: list          = Field(description="All document chunks")
    k: int              = Field(default=TOP_K, description="Number of results")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:

        # ── BM25 retrieval ────────────────────────────────────────────
        # tokenize query into individual words
        query_tokens = query.lower().split()

        # get BM25 scores for all chunks
        bm25_scores = self.bm25.get_scores(query_tokens)

        # get indices of top-k chunks sorted by BM25 score
        top_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:self.k]

        bm25_results = [self.docs[i] for i in top_bm25_indices]

        # ── FAISS retrieval ───────────────────────────────────────────
        faiss_results = self.vectorstore.similarity_search(query, k=self.k)

        # ── RRF combination ───────────────────────────────────────────
        combined = reciprocal_rank_fusion([bm25_results, faiss_results])

        return combined[:self.k]


def build_chain(file_path):

    # ── Step 1: Load PDF ──────────────────────────────────────────────────────
    raw_text = load_pdf(file_path)

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

    # ── Step 3: Build FAISS index ─────────────────────────────────────────────
    vectorstore = FAISS.from_documents(docs, EMBEDDINGS)

    # ── Step 4: Build BM25 index ──────────────────────────────────────────────
    # BM25 needs tokenized text - split each chunk into words
    # This is the "corpus" BM25 searches over
    tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)

    # ── Step 5: Create hybrid retriever ──────────────────────────────────────
    # Combines FAISS and BM25 using RRF
    retriever = HybridRetriever(
        vectorstore=vectorstore,
        bm25=bm25,
        docs=docs,
        k=TOP_K
    )

    # ── Step 6: LLM ───────────────────────────────────────────────────────────
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
    )

    # ── Step 7: Prompt ────────────────────────────────────────────────────────
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

    # ── Helper ────────────────────────────────────────────────────────────────
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # ── Step 8: Build LCEL chain ──────────────────────────────────────────────
    chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # store both indexes on chain for tracing
    chain._vectorstore = vectorstore
    chain._bm25        = bm25
    chain._docs        = docs

    return chain


def traced_query(chain, query: str) -> str:
    trace = Trace(query)

    try:
        # ── Span 1: hybrid retrieval ──────────────────────────────────
        with trace.span("retrieval") as r_span:
            # run BM25
            query_tokens     = query.lower().split()
            bm25_scores      = chain._bm25.get_scores(query_tokens)
            top_bm25_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True
            )[:TOP_K]
            bm25_results = [chain._docs[i] for i in top_bm25_indices]

            # run FAISS with scores for threshold check
            docs_with_scores = chain._vectorstore.similarity_search_with_score(
                query, k=TOP_K
            )
            faiss_results = [doc for doc, score in docs_with_scores]
            top_score     = round(float(docs_with_scores[0][1]), 4) if docs_with_scores else None

            # combine with RRF
            combined         = reciprocal_rank_fusion([bm25_results, faiss_results])
            chunks_retrieved = len(combined)

            r_span.metadata["chunks_retrieved"] = chunks_retrieved
            r_span.metadata["top_score"]        = top_score
            r_span.metadata["retrieval_method"] = "hybrid_bm25_faiss_rrf"

        # ── Check 1: did retrieval find anything? ─────────────────────
        if chunks_retrieved < MIN_CHUNKS_REQUIRED:
            msg = "I couldn't find any relevant sections in the document for your question."
            trace.finish(
                answer     = msg,
                error      = "retrieval returned 0 chunks",
                error_type = ErrorType.RETRIEVAL_FAILURE
            )
            return msg

        # ── Check 2: are the chunks relevant enough? ──────────────────
        if top_score and top_score > RETRIEVAL_SCORE_THRESHOLD:
            msg = "Your question doesn't seem related to the document. Please ask something about the uploaded PDF."
            print(f"  [WARN] Low retrieval score: {top_score} - blocking LLM call")
            trace.finish(
                answer     = msg,
                error      = f"low retrieval score: {top_score}",
                error_type = ErrorType.RETRIEVAL_FAILURE
            )
            return msg

        # ── Span 2: LLM ───────────────────────────────────────────────
        with trace.span("llm") as llm_span:
            answer = chain.invoke(query)

            if not answer or len(answer.strip()) == 0:
                raise ValueError("LLM returned empty response")

            llm_span.metadata["token_estimate"] = len(answer) // 4

        trace.finish(answer=answer)
        return answer

    except ValueError as e:
        msg = "The model returned an empty response. Please try again."
        trace.finish(error=str(e), error_type=ErrorType.LLM_FAILURE)
        return msg

    except Exception as e:
        error_type = classify_error(e)
        print(f"\n  [ERROR] {error_type}: {str(e)}")
        messages = {
            ErrorType.API_ERROR:      "The AI service is temporarily unavailable. Please try again in a moment.",
            ErrorType.LLM_FAILURE:    "The model took too long to respond. Please try again.",
            ErrorType.EMPTY_DOCUMENT: "The document appears to have no readable text.",
            ErrorType.UNKNOWN:        "Something went wrong. Please try again.",
        }
        msg = messages.get(error_type, messages[ErrorType.UNKNOWN])
        trace.finish(error=str(e), error_type=error_type)
        return msg