import os
from LLM import OllamaLLM
from typing import Iterable, List

from embeding import Embedding


def enumerate_files(directory: str, extensions: List[str]) -> Iterable[str]:
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        # if filename ends with one of the extensions
        if any([filename.endswith(ext) for ext in extensions]):
            yield full_path
        elif os.path.isdir(full_path):
            yield from enumerate_files(full_path, extensions)

def process_files(directory: str, extensions: List[str], embedding: Embedding):

    for file_for_processing in enumerate_files(directory, extensions):
        processed_files = embedding.load_content_from_path(file_for_processing)
        print(f"{file_for_processing} has been processed.") if len(processed_files) > 0 else print(f"{file_for_processing} has NOT been processed.")
    

if __name__ == "__main__":
    embedding = Embedding()
    ollama = OllamaLLM(embedding)
    wiki_path = "/Users/andriikozin/prj/ms/wiki"
    tcs_wiki = "IdentityWiki/Services/Routing/TCS-(Traffic-Control-Service)"
    
    markdown_directory = '/'.join([wiki_path, tcs_wiki])
    
    if embedding.is_vectorstore_empty():
        print("Vectorstore is empty. Loading markdown files.")
        process_files(markdown_directory, ['md'], embedding)
    
    # Enter into an interactive loop for conversation
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting the conversation.")
            break
        response = ollama.talk(prompt)
        print(f"Ollama: {response}")