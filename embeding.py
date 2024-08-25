from typing import List
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


class Embedding:
    def __init__(self, 
                ollama_base_url: str = "http://localhost:11434",
                ollama_model: str = "nomic-embed-text", 
                chroma_db_name: str = "chroma_db",
                chroma_db_path: str = "./chroma_db",
                text_splitter_chunk_size: int = 1000,
                text_splitter_chunk_overlap: int = 200,
                debug: bool = False) -> None:
        self._debug = debug
        self._oembed = OllamaEmbeddings(base_url=ollama_base_url, model=ollama_model)
        self._vectorstore = Chroma(
            chroma_db_name, 
            self._oembed,
            chroma_db_path)
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=text_splitter_chunk_size,
            chunk_overlap=text_splitter_chunk_overlap)
    
    @property
    def vectorstore(self):
        return self._vectorstore
        
    async def aload_content_from_path(self, file_path) -> List[str]:
        print(f"Loading content of file: {file_path}") if self._debug else None
        
        loader = TextLoader(file_path)
        docs = await loader.aload()
        if not docs or len(docs) == 0:
            print(f"{file_path}. Skipping, empty.") if self._debug else None
            return []

        splits = self._text_splitter.split_documents(docs)
        if splits is None or len(splits) == 0:
            print(f"{file_path}. Skipping. No splits to add.") if self._debug else None
            return []
        
        return await self._vectorstore.aadd_documents(splits)
    
    def load_content_from_path(self, file_path) -> List[str]:
        print(f"Loading content of file: {file_path}") if self._debug else None
        
        loader = TextLoader(file_path)
        docs = loader.load()
        if not docs or len(docs) == 0:
            print(f"{file_path}. Skipping, empty.") if self._debug else None
            return []

        splits = self._text_splitter.split_documents(docs)
        if splits is None or len(splits) == 0:
            print(f"{file_path}. Skipping. No splits to add.") if self._debug else None
            return []
        
        return self._vectorstore.add_documents(splits)

    def is_vectorstore_empty(self) -> bool:
        if self._vectorstore is None:
            return True
        # Check if the vectorstore has any documents
        doc_count = self._vectorstore._collection.count()  # Assuming Chroma uses a collection with a count method
        return doc_count == 0