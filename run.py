import os
from LLM import OllamaLLM
from typing import List, Callable

def analyze_markdown(file_content, cb: Callable[[str], List]) -> List:
    
    response = cb(file_content)
    
    return response

# write a function which goes through the provide directory and returns a list of markdown files
# it also should go recursively through the subdirectories

def get_markdown_files(directory):
    markdown_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            markdown_files.append(os.path.join(directory, filename))
        elif os.path.isdir(os.path.join(directory, filename)):
            markdown_files.extend(get_markdown_files(os.path.join(directory, filename)))
    return markdown_files


def process_markdown_files(directory, cb: Callable[[str], List]):
    all_docs = []

    for filename in get_markdown_files(directory):
        all_docs.extend(analyze_markdown(filename, cb))
        # Here you would process the result to reorganize or clean the data
        print(f"Processed {filename}")
    return all_docs

if __name__ == "__main__":
    ollama = OllamaLLM()
    wiki_path = "/Users/andriikozin/prj/ms/wiki"
    tcs_wiki = "IdentityWiki/Services/Routing/TCS-(Traffic-Control-Service)"
    markdown_directory = '/'.join([wiki_path, tcs_wiki])
    all_docs = process_markdown_files(markdown_directory, ollama.load_markdown)
    ollama.submit_to_model(all_docs)
    
    # Enter into an interactive loop for conversation
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting the conversation.")
            break
        response = ollama.talk(prompt)
        print(f"Ollama: {response}")