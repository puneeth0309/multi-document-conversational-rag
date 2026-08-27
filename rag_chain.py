from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda

from sentence_transformers import CrossEncoder

from dotenv import load_dotenv

load_dotenv()

# ── Load heavy models ONCE at module level (thread-safe) ─
# Avoids st.cache_resource which breaks inside background threads
print("Loading reranker model...")
_RERANKER = CrossEncoder("BAAI/bge-reranker-base")

print("Loading embedding model...")
_EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Models loaded.")


# ── Reranker ─────────────────────────────────────────────
def rerank_documents(query: str, docs: list, top_k: int = 5) -> list:
    pairs = [(query, doc.page_content) for doc in docs]
    scores = _RERANKER.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


def create_rerank_retriever(base_retriever):
    def rerank_fn(query: str):
        docs = base_retriever.invoke(query)
        return rerank_documents(query, docs) if docs else docs
    return RunnableLambda(rerank_fn)


# ── Main chain loader ─────────────────────────────────────
def load_qa_chain(chat_history: list, mode: str = "smart"):

    vectordb = Chroma(
        persist_directory="./chroma_db",
        embedding_function=_EMBEDDINGS
    )

    base_retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 20, "fetch_k": 40}
    )

    retriever = create_rerank_retriever(base_retriever)

    # ── Build chat history ────────────────────────────────
    history = []
    for msg in chat_history:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))

    # ── LLM ───────────────────────────────────────────────
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)

    # ── History-aware query rewriter ──────────────────────
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Rewrite the user's question into a clear standalone question "
            "using chat history if needed. Return only the rewritten question."
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # ── System prompt ─────────────────────────────────────
    if mode == "strict":
        system_prompt = (
            "You are a document assistant. Answer ONLY using the context below.\n"
            "If the answer is not present, say: "
            "\"This information is not in the uploaded document.\"\n\n"
            "Query type: {query_type}\n\n"
            "{format_instructions}\n\n"
            "Context:\n{context}"
        )
    else:
        system_prompt = (
            "You are a helpful document assistant. "
            "Use the context to answer accurately.\n"
            "If the context is insufficient, you may use general knowledge "
            "but flag it clearly with: \"(Based on general knowledge)\"\n\n"
            "Query type: {query_type}\n\n"
            "{format_instructions}\n\n"
            "Context:\n{context}"
        )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain, retriever, history