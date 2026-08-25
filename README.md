# RAG Pipeline — PDF & Text Q&A with Source Citations

A retrieval-augmented generation (RAG) pipeline that answers questions grounded in a set of source documents — three research papers(PDFs) ("Attention Is All You Need," BERT, and ResNet) and 2 short text files — with cited sources, confidence scores, and conversation history.

## How it works

**Ingestion (run once):**
1. **Load** — PDFs and text files are read via LangChain's `PyPDFLoader`/`TextLoader`, tagged with source metadata (filename, page number).
2. **Chunk** — Documents are split into smaller passages with `RecursiveCharacterTextSplitter` for focused, retrievable pieces.
3. **Embed** — Each chunk is converted into a vector using `sentence-transformers/all-MiniLM-L6-v2`, running locally — no API calls, no cost.
4. **Store** — Vectors are persisted in a local **ChromaDB** vector store.

**Query (run per question):**
5. **Embedding** - A user's question is embedded with the same MiniLM model.
6. **Retrieval** - The vector store returns the most similar chunks via cosine similarity, filtered by a minimum relevance score.
7. **Augmentation** - Retrieved chunks are assembled into a prompt alongside the question.
8. **Groq generation** (`openai/gpt-oss-20b`) generates a grounded answer, returned with source citations, a confidence score, and running conversation history.

## Stack

- **Framework:** LangChain
- **Embeddings:** `sentence-transformers` (local, free)
- **Vector store:** ChromaDB
- **LLM:** Groq API
- **Environment:** `uv`

## Setup

```bash
uv venv
source .venv/bin/activate
uv add -r requirements.txt
```

Create a `.env` file in the project root:
