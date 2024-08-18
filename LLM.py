from typing import Iterable
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


class OllamaLLM:
    def __init__(self, model_name='gemma2:2b'):
        self.model = Ollama(
            base_url='http://localhost:11434',
            model=model_name
        )
        self.oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
        self.vectorstore = None
    
    def load_markdown(self, file_path) -> Iterable[Document]:
        print(f"Loading markdown file: {file_path}")
        loader = TextLoader(
            file_path
        )
        data = loader.load()
        for doc in data:
            yield doc
        
    def submit_to_model(self, docs: Iterable[Document]) -> None:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        del self.vectorstore
        self.vectorstore = Chroma.from_documents(documents=all_splits, embedding=self.oembed)
    
    def talk(self, prompt):
        if self.vectorstore is None:
            return "Nothing to talk about. Please load some documents first."
        
        docs = self.vectorstore.similarity_search(prompt)
        print(len(docs))
        qachain=RetrievalQA.from_chain_type(self.model, retriever=self.vectorstore.as_retriever())
        res = qachain.invoke({"query": prompt})
        return res['result']

if __name__ == "__main__":

    ollama_llm = OllamaLLM()
    response = ollama_llm.talk("What is the capital of France?")
    print(response)


# import nltk
# nltk.download('wordnet')