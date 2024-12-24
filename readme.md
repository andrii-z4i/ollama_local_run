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

Create a virtual environment
-----

1. `python -m venv .venv`
1. Activate it `source ./.venv/bin/activate . ` or on Windows`./.venv/Scripts/activate.ps1`
1. Install dependencies `pip install -r requirements.txt`

How to kill db 
----

on Windows
`remove-Item -Recurse -Force .\chroma_db\`

on Linux
`rm -rf ./chroma_db/`

How to run?
---

System contains of two running modules:
1. `Embedding Files processing` which is responsible for enumeration of files in directories and preparing the binary representation of data (embedding)
1. `Chat` which is responsible for connecting the prepared embedding for actual conversation based on it.

To run the #1 
`python .\embedding_files_processor  --directory-to-analyze 'some/path/to/files' --extensions md --verbose --reload`

To run the #2
`python .\chat.py --system-prompt "I'm the customer of the Super Nice system"`

More information about embedding logic in [UpdateEmbeddings](./docs/updateEmbeddings.md)