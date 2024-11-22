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

Help
`python .\run.py --help`

Example of real run
`python .\run.py --directory-to-analyze ..\Services\Routing  --extensions md --verbose --system-prompt "You are the engineer on the routing space where Gateway and Traffic Control services are developed"`
