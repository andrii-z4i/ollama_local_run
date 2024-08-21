import os
from LLM import OllamaLLM
from typing import Iterable


def get_markdown_files(directory) -> Iterable[str]:
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        if filename.endswith(".md"):
            yield full_path
        elif os.path.isdir(full_path):
            yield from get_markdown_files(full_path)

def process_markdown_files(directory, llm: OllamaLLM):

    for md_file_path in get_markdown_files(directory):
        processed_md = llm.load_markdown(md_file_path)
        llm.submit_to_model(processed_md)
        print(f"Processed {md_file_path}")
    

if __name__ == "__main__":
    ollama = OllamaLLM()
    wiki_path = "/Users/andriikozin/prj/ms/wiki"
    tcs_wiki = "IdentityWiki/Services/Routing/TCS-(Traffic-Control-Service)"
    
    markdown_directory = '/'.join([wiki_path, tcs_wiki])
    all_docs = process_markdown_files(markdown_directory, ollama)
    
    # Enter into an interactive loop for conversation
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting the conversation.")
            break
        response = ollama.talk(prompt)
        print(f"Ollama: {response}")