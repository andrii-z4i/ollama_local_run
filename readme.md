Installation process
=======


`winget install python3` on Windows

Download and install [ollama](https://ollama.com/).

Configure ollama
-----

```
ollama run nomic-embed-text
ollama run gemma2:2b
```

For **photo analysis**, also pull a vision model:

```
ollama run gemma3
```

Create a virtual environment
-----

1. `python -m venv .venv`  (requires Python 3.10+)
1. Activate it `source ./.venv/bin/activate` or on Windows `./.venv/Scripts/activate.ps1`
1. Install dependencies `pip install -r requirements.txt`

How to kill db 
----

on Windows
`remove-Item -Recurse -Force .\chroma_db\`

on Linux
`rm -rf ./chroma_db/`

How to run?
---

This project has three modes of operation:

### 1. Text Embedding Pipeline

Enumerates files in a directory and stores their embeddings in ChromaDB.

```
python embedding_files_processor.py --directory-to-analyze 'some/path/to/files' --extensions md --verbose --reload
```

### 2. Text Chat

Interactive conversation against the text embeddings.

```
python chat.py --system-prompt "I'm the customer of the Super Nice system"
```

### 3. Photo Analysis Pipeline (NEW)

Analyzes photos with a vision model, extracts EXIF metadata (dates, GPS, camera), reverse-geocodes locations, and stores everything in ChromaDB.

```
python analyze_photos.py --directory-to-analyze '/path/to/photos' --verbose --reload
```

Key flags:
- `--vision-model gemma3` — Ollama vision model (default: gemma3)
- `--extensions jpg jpeg png heic webp` — image formats (these are the defaults)
- `--exclude-subdirectories Thumbnails .thumbnails` — skip specific subfolders
- `--geocode-cache .geocode_cache.json` — cache for reverse geocoding results
- `--reload` — re-analyze photos whose content changed
- `--chroma-db-name photo_db` — collection name (default: photo_db)

### 4. Photo Memory MCP Server (NEW)

Exposes the photo database as an MCP server for use with GitHub Copilot or any MCP-compatible client.

```
python mcp_server.py --chroma-db-path ./chroma_db --chroma-db-name photo_db
```

#### MCP Tools Available

| Tool | Description |
|------|-------------|
| `search_photos` | Semantic search by description (e.g., "sunset on a beach") |
| `search_photos_by_date` | Find photos by date or date range |
| `search_photos_by_location` | Find photos by location name |
| `get_photo_details` | Get full metadata + description for a specific photo |
| `get_photo_timeline` | Chronological summary of photos in a date range |

#### Copilot MCP Configuration

Add to your VS Code `settings.json` or `.vscode/mcp.json`:

```json
{
  "mcp": {
    "servers": {
      "photo-memory": {
        "command": "python",
        "args": ["mcp_server.py", "--chroma-db-path", "./chroma_db"],
        "cwd": "/path/to/ollama_local_run"
      }
    }
  }
}
```

More information about embedding logic in [UpdateEmbeddings](./docs/updateEmbeddings.md)