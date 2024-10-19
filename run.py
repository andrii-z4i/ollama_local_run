import os
from LLM import OllamaLLM
from typing import Iterable, List
from embeding import Embedding
import asyncio
from run_arguments import RunArguments


def enumerate_files(directory: str, extensions: List[str], exclude_subfolders: List[str],) -> Iterable[str]:
    for filename in os.listdir(directory):
        if filename in exclude_subfolders:
            continue
        full_path = os.path.join(directory, filename)
        # if filename ends with one of the extensions
        if any([filename.endswith(ext) for ext in extensions]):
            yield full_path
        elif os.path.isdir(full_path):
            yield from enumerate_files(full_path, extensions, exclude_subfolders)

async def process_files(directory: str, extensions: List[str], exclude_subfolders: List[str], embedding: Embedding):
    tasks = []
    for file_for_processing in enumerate_files(directory, extensions, exclude_subfolders):
        tasks.append(embedding.aload_content_from_path(file_for_processing))
    
    await asyncio.gather(*tasks)
        
    

if __name__ == "__main__":

    run_args = RunArguments().parse()

    embedding = Embedding(
        chroma_db_name=run_args.chroma_db_name,
        chroma_db_path=run_args.chroma_db_path,
        debug=run_args.verbose
        )
    ollama = OllamaLLM(embedding, system_prompt=run_args.system_prompt)
    
    if any([embedding.is_vectorstore_empty(), run_args.force_reload, run_args.soft_reload]):
        print("Vectorstore is empty. Loading markdown files.")
        asyncio.run(process_files(run_args.directory_to_analyze, run_args.extensions, run_args.exclude_subdirectories, embedding))
    
    # Enter into an interactive loop for conversation
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting the conversation.")
            break
        response = ollama.talk(prompt)
        print(f"Ollama: {response}")