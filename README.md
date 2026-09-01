# 🚀 Multi-Document Conversational RAG System

A Streamlit-based Retrieval-Augmented Generation (RAG) application that allows users to upload multiple PDF documents and ask conversational questions about their content.

The system combines semantic retrieval, Maximal Marginal Relevance (MMR), cross-encoder reranking, history-aware query rewriting, and a Groq-hosted LLM to generate context-aware answers with document source references.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/LangChain-RAG-green" alt="LangChain">
  <img src="https://img.shields.io/badge/Groq-GPT--OSS--20B-orange" alt="Groq">
  <img src="https://img.shields.io/badge/Vector_DB-ChromaDB-purple" alt="ChromaDB">
</p>

---


## ✨ Features

- ➜ **Multi-PDF Support** — upload and index multiple PDF documents
- ➜ **Conversational RAG** — ask follow-up questions using chat history
- ➜ **History-Aware Query Rewriting** — converts follow-up questions into standalone queries
- ➜ **ChromaDB Vector Storage** — stores document chunks and embeddings
- ➜ **Hugging Face Embeddings** — uses `all-MiniLM-L6-v2`
- ➜ **MMR Retrieval** — retrieves relevant and diverse document chunks
- ➜ **BGE Reranking** — uses `BAAI/bge-reranker-base` for cross-encoder reranking
- ➜ **Smart & Strict Modes** — choose flexible or document-only answering
- ➜ **Streaming Responses** — streams LLM output through Groq
- ➜ **Query-Type Formatting** — formats compare, summarize, definition, explain, and general queries
- ➜ **Source Attribution** — shows file and page-level references
- ➜ **Chat Export** — download the conversation as `.txt`
- ➜ **Streamlit UI** — simple interactive web interface

## 🏗️ Architecture

```text
                     MULTI-DOCUMENT CONVERSATIONAL RAG
                                   │
                                   ▼
                            ┌──────────────┐
                            │ Streamlit UI │
                            │    app.py    │
                            └──────┬───────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
           DOCUMENT INGESTION               QUERY PIPELINE
                    │                             │
                    ▼                             ▼
              Upload PDFs                  User Question
                    │                             │
                    ▼                             ▼
              PyPDFLoader                   Chat History
                    │                             │
                    ▼                             ▼
             Extract Text             History-Aware Rewriting
                    │                             │
                    ▼                             ▼
       RecursiveCharacterTextSplitter      Standalone Query
        chunk=800 / overlap=150                  │
                    │                             ▼
                    ▼                    all-MiniLM-L6-v2
           all-MiniLM-L6-v2               Query Embedding
          Document Embeddings                    │
                    │                             ▼
                    └──────────────► ChromaDB / MMR
                                           │
                                           ▼
                                  fetch_k=40 → k=20
                                           │
                                           ▼
                                      BGE Reranker
                                           │
                                           ▼
                                      Top 5 Chunks
                                           │
                                           ▼
                                   Smart / Strict Prompt
                                           │
                                           ▼
                                    Groq-hosted LLM
                                   openai/gpt-oss-20b
                                           │
                                           ▼
                                  Streaming Answer
                                           │
                                           ▼
                                    Source References
```

## 🔄 How It Works

### ➜ 1. Document Ingestion

When a user uploads a PDF, the application temporarily stores the file and sends it to `ingest.py`.

The ingestion pipeline:

```text
PDF
 ↓
PyPDFLoader
 ↓
Extracted Documents
 ↓
RecursiveCharacterTextSplitter
 ↓
Text Chunks
 ↓
all-MiniLM-L6-v2
 ↓
384-dimensional Embeddings
 ↓
ChromaDB
```

The current text splitting configuration is:

```python
RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
```

The embedding model is initialized once at module level and reused:

```python
HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
```

### ➜ 2. Query Processing

When the user asks a question, `app.py` calls:

```python
load_qa_chain(
    st.session_state.messages,
    mode=mode
)
```

The query pipeline is created in `rag_chain.py`.

#### ↳ History-Aware Query Rewriting

Follow-up questions may depend on previous conversation context.

Example:

```text
User: What is diabetes?
Assistant: ...

User: What are its symptoms?
```

Before retrieval, the LLM can rewrite the follow-up into a standalone query such as:

```text
What are the symptoms of diabetes?
```

This is implemented using:

```python
create_history_aware_retriever(...)
```

with the conversation history supplied through:

```python
MessagesPlaceholder("chat_history")
```

### ➜ 3. MMR Retrieval

The query is embedded using `all-MiniLM-L6-v2` and searched against ChromaDB.

```python
base_retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 20,
        "fetch_k": 40
    }
)
```

MMR, or Maximal Marginal Relevance, balances:

- relevance to the user query
- diversity among retrieved chunks

This helps reduce repetitive context.

### ➜ 4. Cross-Encoder Reranking

The chunks retrieved through MMR are then reranked using:

```text
BAAI/bge-reranker-base
```

The reranker evaluates each query-document pair directly and assigns a relevance score.

```text
Query + Chunk 1 → Score
Query + Chunk 2 → Score
Query + Chunk 3 → Score
...
```

The results are sorted by score, and the top 5 chunks are kept as context for answer generation.

### ➜ 5. Answer Generation

The final retrieved chunks are inserted into the prompt using:

```python
create_stuff_documents_chain(...)
```

The complete retrieval and generation pipeline is connected using:

```python
create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain
)
```

The application currently uses:

```python
ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2
)
```

Responses are streamed back to the Streamlit interface.

## 🎯 Smart and Strict Modes

### ➜ Smart Mode

Uses retrieved document context first. If the document context is insufficient, the model may use general knowledge and clearly label it as:

```text
(Based on general knowledge)
```

### ➜ Strict Mode

The model is instructed to answer only using information retrieved from the uploaded documents.

If the answer is unavailable, it responds with:

```text
This information is not in the uploaded document.
```

## 🧠 Query-Type Formatting

The application uses simple rule-based query classification to identify:

- comparison
- summarization
- definition
- explanation
- general questions

Different formatting instructions are then supplied to the LLM to improve answer readability.

## 📁 Project Structure

```text
multi-document-conversational-rag/
│
├── app.py
├── ingest.py
├── rag_chain.py
├── requirements.txt
├── runtime.txt
├── .gitignore
├── .env
└── README.md
```

### `app.py`

Handles:

- Streamlit UI
- PDF uploads
- Smart/Strict mode selection
- session state
- chat history
- query classification
- response streaming
- source display
- chat export

### `ingest.py`

Handles:

- PDF loading
- text extraction
- text splitting
- embedding generation
- ChromaDB indexing

### `rag_chain.py`

Handles:

- ChromaDB connection
- query embedding
- MMR retrieval
- BGE reranking
- chat-history conversion
- history-aware query rewriting
- prompt creation
- Groq LLM integration
- final RAG chain construction

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| RAG Framework | LangChain |
| PDF Loader | PyPDFLoader |
| Embeddings | Hugging Face `all-MiniLM-L6-v2` |
| Vector Database | ChromaDB |
| Retrieval | MMR |
| Reranking | `BAAI/bge-reranker-base` |
| LLM Provider | Groq |
| LLM | `openai/gpt-oss-20b` |
| Language | Python |

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/puneeth0309/multi-document-conversational-rag.git
cd multi-document-conversational-rag
```

Create and activate a Conda environment:

```bash
conda create -n advance_rag_chatbot python=3.10
conda activate advance_rag_chatbot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not commit your `.env` file to GitHub.

Recommended `.gitignore`:

```gitignore
.env
chroma_db/
__pycache__/
*.pyc
```

## ▶️ Run the Application

Use Streamlit to start the project:

```bash
streamlit run app.py
```

Do not run the project with:

```bash
python app.py
```

because Streamlit features such as session state and chat components require the Streamlit runtime.

## 💬 Example Conversation

```text
User:
What is diabetes?

Assistant:
Diabetes is ...

User:
What are its symptoms?

History-aware rewritten query:
What are the symptoms of diabetes?

Retrieval:
ChromaDB → MMR → BGE reranking

Assistant:
The common symptoms include ...
```

## 🔁 Key RAG Flow

```text
Upload PDF
   ↓
Extract Text
   ↓
Chunk Documents
   ↓
Generate Embeddings
   ↓
Store in ChromaDB


User Question
   ↓
Use Chat History
   ↓
Rewrite Follow-Up Query if Required
   ↓
Generate Query Embedding
   ↓
ChromaDB + MMR Retrieval
   ↓
BGE Reranking
   ↓
Top Relevant Context
   ↓
Prompt + History + Question
   ↓
Groq LLM
   ↓
Streaming Answer + Sources
```

## 🚧 Future Improvements

Possible improvements include:

- content-hash based duplicate document detection
- background or asynchronous document indexing
- improved handling of scanned PDFs using OCR
- retrieval relevance thresholding
- document-level filtering
- better handling of ambiguous follow-up questions
- caching and performance optimization
- using the exact retrieved context returned by the RAG chain for source attribution
- support for additional document formats such as TXT and DOCX

