# Copilot Instructions

## Prerequisites

This project requires a running [Ollama](https://ollama.com/) instance with two models pulled:

- `nomic-embed-text` — used for generating embeddings
- `gemma2:2b` — used as the chat LLM

Ollama must be accessible at `http://localhost:11434`.

## Setup

```sh
python -m venv .venv
source .venv/bin/activate   # Windows: .venv/Scripts/activate.ps1
pip install -r requirements.txt
```

## How to run

**Step 1 — Build embeddings** from a directory of files:

```sh
python embedding_files_processor.py --directory-to-analyze '/path/to/files' --extensions md --verbose --reload
```

Key flags: `--reload` re-embeds files whose checksum changed; `--exclude-subdirectories` skips named subfolders.

**Step 2 — Chat** against the embedded data:

```sh
python chat.py --system-prompt "I'm the customer of the Super Nice system"
```

Both commands share `--chroma-db-name` and `--chroma-db-path` flags (default: `chroma_db` / `./chroma_db`).

To reset the vector database: `rm -rf ./chroma_db/`

## Architecture

This is a two-stage RAG (Retrieval-Augmented Generation) application built on LangChain and ChromaDB, running entirely against a local Ollama instance.

### Stage 1: Embedding pipeline (`embedding_files_processor.py`)

`FilesProcessor` walks a directory, filters by extension, computes SHA-256 checksums, and hands files to `EmbeddingManager`. The embedding manager uses `RecursiveCharacterTextSplitter` (1000 chars / 200 overlap), stores chunks in ChromaDB with `source` and `checksum` metadata, and supports incremental updates — only files with changed checksums are re-embedded. Files removed from disk are also deleted from the vector store.

### Stage 2: Chat (`chat.py`)

`OllamaLLM` connects to the same ChromaDB collection, creates a LangChain retrieval chain (`create_retrieval_chain` + `create_stuff_documents_chain`), and runs an interactive REPL. The system prompt is injected alongside retrieved context.

### Module layout

- `src/embeding_manager.py` — ChromaDB vector store wrapper (CRUD, checksum-based diffing)
- `src/LLM.py` — RAG chain setup and `talk()` method
- `src/files_processor.py` — directory walker, checksum calculator, embedding orchestrator
- `src/arguments/` — argparse wrappers; each entry point has its own `RunArguments` class

## Conventions

- **Argument parsing pattern**: each entry point has a paired module under `src/arguments/` containing a data class (e.g., `ChatRunArguments`) and a `RunArguments` parser class. The data class wraps `argparse.Namespace` with `@property` accessors.
- **No test suite or linter** is configured in this project.
- **ChromaDB data directory** (`chroma_db/`) is gitignored. Never commit it.
- Dependencies are unpinned in `requirements.txt` (package names only, no versions).
