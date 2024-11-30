import asyncio
from typing import List
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from enum import Enum

class EmbeddingProcessMode(Enum):
    SOFT_RELOAD = 1 # Process file only if checksum differes from the one in the vectorstore
    FORCE_RELOAD = 2 # Process file regardless of the checksum in the vectorstore


class Embedding:
    def __init__(self, 
                ollama_base_url: str = "http://localhost:11434",
                ollama_model: str = "nomic-embed-text", 
                chroma_db_name: str = "chroma_db",
                chroma_db_path: str = "./chroma_db",
                text_splitter_chunk_size: int = 1000,
                text_splitter_chunk_overlap: int = 200,
                vectorstore_connections: int = 10,
                debug: bool = False) -> None:
        self._debug = debug
        self._oembed = OllamaEmbeddings(base_url=ollama_base_url, model=ollama_model)
        
        if vectorstore_connections < 1:
            raise ValueError("Vectorstore connections must be more than 1.")
        
        self._vectorstore = [Chroma(
            chroma_db_name, 
            self._oembed,
            chroma_db_path) for _ in range(vectorstore_connections)]
        
        self._next_vectorstore = 0
        self._text_splitter_chunk_size = text_splitter_chunk_size
        self._text_splitter_chunk_overlap = text_splitter_chunk_overlap
    
    @property
    def vectorstore(self):
        return self._vectorstore[0]
        
    async def _aload_content_from_path(self, file_path: str, checksum: str, process_mode: EmbeddingProcessMode) -> List[str]:
        print(f"Loading content of file: {file_path}") if self._debug else None
        
        if process_mode != EmbeddingProcessMode.FORCE_RELOAD and process_mode != EmbeddingProcessMode.SOFT_RELOAD:
            raise ValueError("Invalid process mode.")

        if process_mode == EmbeddingProcessMode.SOFT_RELOAD:
            # Check if the file has already been processed
            raise NotImplementedError("Soft reload not implemented yet.")
            # If file is in the vectorstore and checksum is the same, skip processing
            if self._vectorstore[0].document_exists(file_path):
                print(f"{file_path}. Skipping, already processed.") if self._debug else None
                return []
        

        loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
        docs = await loader.aload()
        if not docs or len(docs) == 0:
            print(f"{file_path}. Skipping, empty.") if self._debug else None
            return []
        
        _text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._text_splitter_chunk_size,
            chunk_overlap=self._text_splitter_chunk_overlap)

        splits = _text_splitter.split_documents(docs)
        if splits is None or len(splits) == 0:
            print(f"{file_path}. Skipping. No splits to add.") if self._debug else None
            return []
        
        # round robin pick of vectorstore to load balance the connections
        self._next_vectorstore += 1
        self._next_vectorstore = self._next_vectorstore % len(self._vectorstore)
        print(f"----> Adding to vectorstore {self._next_vectorstore}") if self._debug else None
        return await self._vectorstore[self._next_vectorstore].aadd_documents(splits)
    
    async def aload_content_from_path(self, file_path: str, checksum: str, process_mode: EmbeddingProcessMode) -> List[str]:
        retries = 3
        delay = 2 # seconds
        for attempt in range(retries):
            try:
                return await self._aload_content_from_path(file_path, checksum, process_mode)
            except Exception as e:
                print(f"Failed to load content from {file_path}. Retrying...") if self._debug else None
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                else:
                    raise  # Re-raise the last exception if all retries fail
    
    def load_content_from_path(self, file_path) -> List[str]:
        print(f"Loading content of file: {file_path}") if self._debug else None
        
        loader = TextLoader(file_path)
        docs = loader.load()
        if not docs or len(docs) == 0:
            print(f"{file_path}. Skipping, empty.") if self._debug else None
            return []

        _text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._text_splitter_chunk_size,
            chunk_overlap=self._text_splitter_chunk_overlap)
        
        splits = _text_splitter.split_documents(docs)
        if splits is None or len(splits) == 0:
            print(f"{file_path}. Skipping. No splits to add.") if self._debug else None
            return []
        
        return self._vectorstore[0].add_documents(splits)