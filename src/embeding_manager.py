from typing import List
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader


class EmbeddingManager:
    """
    Manages the embedding and vector storage of documents using Ollama and Chroma.
    Attributes:
        _debug (bool): Flag to enable debug mode.
        _oembed (OllamaEmbeddings): Instance of OllamaEmbeddings for embedding operations.
        _vectorstore (Chroma): Instance of Chroma for vector storage operations.
        _text_splitter_chunk_size (int): Size of chunks for text splitting.
        _text_splitter_chunk_overlap (int): Overlap size for text splitting.
    Methods:
        vectorstore: Property to access the vector store.
        find_documents_in_vectorstore(document_path: str) -> List[Document]: Finds documents in the vector store by their path.
        get_list_of_stored_files() -> List[str]: Retrieves a list of stored file paths from the vector store.
        _delete_documents_by_path(document_path: str) -> None: Deletes documents from the vector store by their path.
        _should_files_be_deleted(documents: List[Document], checksum: str, reload: bool) -> bool: Determines if files should be deleted based on checksum and reload flag.
        load_content_from_path(file_path: str, checksum: str, reload: bool) -> List[str]: Loads content from a file path, processes it, and adds it to the vector store.
    """
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
    
    def find_documents_in_vectorstore(self, document_path: str) -> List[Document]:
        """
        Finds documents in the vector store that match the given document path.
        Args:
            document_path (str): The path of the document to search for in the vector store.
        Returns:
            List[Document]: A list of documents that match the filter criteria.
        """
        # Filter the vector store by metadata
        filter_criteria = {"source": document_path }  # Adjust key to match your metadata key
        results = self.vectorstore.search(
            query="dummy",  # Replace with any dummy query text (not used in filtering)
            search_type="similarity",
            filter=filter_criteria
        )
        
        return results
    
    def get_list_of_stored_files(self) -> List[str]:
        """
        Retrieves a list of stored file sources from the vector store.

        This method accesses the vector store's collection, extracts the metadata
        of all documents, and returns a list of unique file sources.

        Returns:
            List[str]: A list of unique file sources stored in the vector store.
        """
        _all_documents = self.vectorstore._collection.get()
        return list(set([metadata["source"] for metadata in _all_documents['metadatas']]))
    
    def _delete_documents_by_path(self, document_path: str) -> None:
        """
        Deletes documents from the vector store based on the provided document path.

        Args:
            document_path (str): The path of the document to be deleted.

        Returns:
            None
        """
        print(f"Deleting documents by path {document_path}") if self._debug else None
        self.vectorstore._collection.delete(where={"source": document_path})
    
    def _should_files_be_deleted(self, documents: List[Document], checksum: str, reload: bool) -> bool:
        """
        Determines whether files should be deleted based on the provided documents, checksum, and reload flag.
        Args:
            documents (List[Document]): A list of Document objects to check.
            checksum (str): The checksum value to compare against the documents' metadata.
            reload (bool): A flag indicating whether to force a reload.
        Returns:
            bool: True if files should be deleted, False otherwise.
        """
        if len(documents) == 0:
            return False
        
        if reload == True:
            return True if any([doc.metadata["checksum"] != checksum for doc in documents]) else False
                
        return False

    def load_content_from_path(self, file_path: str, checksum: str, reload: bool) -> List[str]:
        """
        Loads content from the specified file path, processes it, and adds it to the vectorstore if necessary.
        Args:
            file_path (str): The path to the file to be loaded.
            checksum (str): The checksum of the file to verify its integrity.
            reload (bool): Flag indicating whether to reload the file even if it exists in the vectorstore.
        Returns:
            List[str]: A list of document IDs added to the vectorstore, or an empty list if no documents were added.
        """
        
        _documents = self.find_documents_in_vectorstore(file_path)
        reload_after_delete = False
        if self._should_files_be_deleted(_documents, checksum, reload):
            self._delete_documents_by_path(file_path)
            reload_after_delete = True
        
        
        if not reload_after_delete and len(_documents) > 0:
            print(f"Skipping. Document already in vectorstore. {file_path}") if self._debug else None
            return []

        loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
        docs = loader.load()
        if not docs or len(docs) == 0:
            print(f"Skipping. Empty file. {file_path}") if self._debug else None
            return []
        
        # enrich documents with metadata
        for doc in docs:
            doc.metadata = {"source": file_path, "checksum": checksum}
        
        _text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._text_splitter_chunk_size,
            chunk_overlap=self._text_splitter_chunk_overlap)

        splits = _text_splitter.split_documents(docs)
        if splits is None or len(splits) == 0:
            print(f"Skipping. No splits to add. {file_path}") if self._debug else None
            return []
        
        print(f"----> Adding to vectorstore content of the file {file_path}") if self._debug else None
        return self._vectorstore.add_documents(splits)
