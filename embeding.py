import asyncio
from typing import List
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from enum import Enum


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
        self._vectorstore = Chroma( chroma_db_name, self._oembed, chroma_db_path) 
        self._text_splitter_chunk_size = text_splitter_chunk_size
        self._text_splitter_chunk_overlap = text_splitter_chunk_overlap
    
    @property
    def vectorstore(self):
        return self._vectorstore
    
    def find_documents_in_vectorstore(self, document_path: str) -> list[Document]:
        """
        Check if a specific document (by path) is in the Chroma vector store with metadata filtering.
        
        Args:
            vectorstore (Chroma): The Chroma vector store instance.
            document_path (str): The path of the document to check.
            
        Returns:
            bool: True if the document is in the vector store, False otherwise.
        """
        # Filter the vector store by metadata
        filter_criteria = {"source": document_path }  # Adjust key to match your metadata key
        results = self.vectorstore.search(
            query="dummy",  # Replace with any dummy query text (not used in filtering)
            search_type="similarity",
            filter=filter_criteria
        )
        
        # Check if any results are returned
        return results
    
    def _delete_documents_by_path(self, document_path: str) -> None:
        print(f"Deleting documents by path {document_path}") if self._debug else None
        self.vectorstore._collection.delete(where={"source": document_path})
    
    def _should_files_be_deleted(self, documents: List[Document], checksum: str, reload: bool) -> bool:
        if len(documents) == 0:
            return False
        
        if reload == True:
            return True if any([doc.metadata["checksum"] != checksum for doc in documents]) else False
                
        return False

    def load_content_from_path(self, file_path: str, checksum: str, reload: bool) -> List[str]:
        
        _documents = self.find_documents_in_vectorstore(file_path)
        reload_after_delete = False
        if self._should_files_be_deleted(_documents, checksum, reload):
            self._delete_documents_by_path(file_path)
            reload_after_delete = True
        
        
        if not reload_after_delete and len(_documents) > 0:
            print(f"{file_path}. Skipping. Document already in vectorstore.") if self._debug else None
            return []

        loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
        docs = loader.load()
        if not docs or len(docs) == 0:
            print(f"{file_path}. Skipping, empty.") if self._debug else None
            return []
        
        # enrich documents with metadata
        for doc in docs:
            doc.metadata = {"source": file_path, "checksum": checksum}
        
        _text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._text_splitter_chunk_size,
            chunk_overlap=self._text_splitter_chunk_overlap)

        splits = _text_splitter.split_documents(docs)
        if splits is None or len(splits) == 0:
            print(f"{file_path}. Skipping. No splits to add.") if self._debug else None
            return []
        
        print(f"----> Adding to vectorstore content of the file {file_path}") if self._debug else None
        return self._vectorstore.add_documents(splits)
