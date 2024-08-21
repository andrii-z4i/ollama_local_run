import os
from typing import Iterable
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


class OllamaLlmSimple:
    def __init__(self, model_name='gemma2:2b'):
        self.model = Ollama(
            base_url='http://localhost:11434',
            model=model_name
        )
        
    
    def load_file_into_model(self, file_path) -> Iterable[Document]:
        print(f"Loading markdown file: {file_path}")
        with open(file_path, 'r') as file:
            content = file.read()
        message = f"file name: {file_path}\ncontent: {content}"
        self.model.invoke(message)
    
    def talk(self, prompt):
        return self.model.invoke(prompt)


def get_markdown_files(directory):
    markdown_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            markdown_files.append(os.path.join(directory, filename))
        elif os.path.isdir(os.path.join(directory, filename)):
            markdown_files.extend(get_markdown_files(os.path.join(directory, filename)))
    return markdown_files


if __name__ == "__main__":

    ollama = OllamaLlmSimple()
    wiki_path = "/Users/andriikozin/prj/ms/wiki"
    tcs_wiki = "IdentityWiki/Services/Routing/TCS-(Traffic-Control-Service)"
    markdown_directory = '/'.join([wiki_path, tcs_wiki])
    all_docs = get_markdown_files(markdown_directory)
    
    for file_path in all_docs:
        ollama.load_file_into_model(file_path)
    
    # Enter into an interactive loop for conversation
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting the conversation.")
            break
        response = ollama.talk(prompt)
        print(f"Ollama: {response}")

