Install
------

python3.x
ollama


```
ollama run nomic-embed-text
ollama run gemma2:2b
python -m venv .venv
```

`source ./venv/bin/activate . `

or 

`./venv/Scripts/activate.ps1`

remove db 
on Windows
`remove-Item -Recurse -Force .\chroma_db\`

on Linux
`rm -rf ./chroma_db/`

how to run?

`python .\run.py --directory-to-analyze C:\Users\z4ian\prj\IdentityWiki.wiki\IdentityWiki\Services\Routing  --extensions md --verbose --system-prompt "You are the engineer on the routing space where Gateway and Traffic Control services are developed"`
