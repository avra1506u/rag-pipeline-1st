# RAG Pipeline — PDF & Text Q&A with Source Citations

A retrieval-augmented generation (RAG) pipeline that answers questions grounded in a set of source documents, with cited sources, confidence scores, and conversation history.

## Corpus

- **23 PDFs total:** the original 3 foundational ML papers ("Attention Is All You Need", BERT, ResNet) plus 20 additional papers sampled from the [Vectara Open RAG Benchmark](https://huggingface.co/datasets/vectara/open_ragbench) (text-only use — see `download_and_filter_corpus.py`).
- **2 short text files.**
- Papers were selected from 200 candidates using a text-density heuristic (chars/page) to favor prose-heavy documents over table/figure-heavy ones — see `corpus_report.txt` for the full kept/dropped breakdown and scores.

## How it works

**Ingestion (run once):**
1. **Load** — PDFs and text files are read via LangChain's `PyPDFLoader`/`TextLoader`, tagged with source metadata (filename, page number).
2. **Chunk** — Documents are split into smaller passages with `RecursiveCharacterTextSplitter` for focused, retrievable pieces.
3. **Embed** — Each chunk is converted into a vector using `sentence-transformers/all-MiniLM-L6-v2`, running locally — no API calls, no cost.
4. **Store** — Vectors are persisted in a local **ChromaDB** vector store.

**Query (run per question):**
1. **Embedding** - A user's question is embedded with the same MiniLM model.
2. **Retrieval** - The vector store returns the most similar chunks via cosine similarity, filtered by a minimum relevance score.
3. **Augmentation** - Retrieved chunks are assembled into a prompt alongside the question.
4. **Groq generation** (`openai/gpt-oss-20b`) generates a grounded answer, returned with source citations, a confidence score, and running conversation history.

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

Create a `.env` file in the project root (never commit this — it's gitignored):

```
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com). Each contributor should use their own key.

**Note:** `data/vector_store/` is gitignored (it exceeds GitHub's 100MB file limit and is fully regenerable). After pulling, re-run the ingestion/embedding cells in `notebook/pdf_loader.ipynb` to build your own local vector store from the PDFs in `data/pdf/`.

## Adding more papers

`download_and_filter_corpus.py` downloads a batch of candidate arXiv papers (by ID) and automatically keeps only the top N most text-dense ones (see script for config). Useful for expanding the corpus without manually checking every PDF for table/figure-heavy content.

## Known limitations

- Retrieval can fail on questions phrased with different terminology than the source text — e.g. "skip connection" doesn't reliably retrieve content that only uses the term "shortcut connection," even at a similarity threshold of 0.0. Some genuinely relevant chunks can score negative similarity.
- The `AdvancedRAGPipeline` class (in `pdf_loader.ipynb`) does not currently return a confidence score — use the `rag_advanced()` function instead when confidence scores are needed.
