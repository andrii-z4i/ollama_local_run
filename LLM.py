from typing import List
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain


class OllamaLLM:
    def __init__(self, model_name='gemma2:2b'):
        self.model = Ollama(
            base_url='http://localhost:11434',
            model=model_name
        )
        self.oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
        self.vectorstore = Chroma(
            "wiki_store", 
            self.oembed,
            "./chroma_langchain_db_yaml")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200)
    
    def load_markdown(self, file_path) -> List[Document]:
        print(f"Loading markdown file: {file_path}")
        loader = TextLoader(
            file_path
        )
        return loader.load()
        
        
    def submit_to_model(self, docs: List[Document]) -> None:
        
        if not docs or len(docs) == 0:
            print(f"---- Empty docs ---------------> No documents to split. Skipping.")
            return

        splits = self.text_splitter.split_documents(docs)
        if splits is None or len(splits) == 0:
            print(f"===================> No splits to submit. Skipping.")
            return
        
        self.vectorstore.add_documents(splits)
    
    def talk(self, human_prompt):
        if self.vectorstore is None:
            return "Nothing to talk about. Please load some documents first."
        
        retriever = self.vectorstore.as_retriever()
        # 2. Incorporate the retriever into a question-answering chain.
        system_prompt = (
            "You are an assistant for Traffic Management Topology definitions. "
            "Use the following pieces of retrieved context to answer "
            "the question. The information you have access to represents the "
            "topology definitions per policy. servicedef.yaml is the Environment, "
            "top level domain, weighted switch definition. While other files are "
            "representing policies and their definitions. "
            " If you don't know the answer, say that you "
            "don't know. Keep the answer concise."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.model, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        result = rag_chain.invoke({"input": human_prompt})
        return result["answer"]

    def is_vectorstore_empty(self) -> bool:
        if self.vectorstore is None:
            return True
        # Check if the vectorstore has any documents
        doc_count = self.vectorstore._collection.count()  # Assuming Chroma uses a collection with a count method
        return doc_count == 0

