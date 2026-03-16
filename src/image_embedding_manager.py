from typing import List, Dict, Any, Optional

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class ImageEmbeddingManager:
    """Manages embedding and storage of image descriptions in ChromaDB.

    Each image is stored as one or more document chunks where:
    - page_content = the vision-model description (used for semantic search)
    - metadata = EXIF data, file path, checksum, location, etc. (used for filtering)
    """

    def __init__(self,
                 ollama_base_url: str = "http://localhost:11434",
                 ollama_model: str = "nomic-embed-text",
                 chroma_db_name: str = "photo_db",
                 chroma_db_path: str = "./chroma_db",
                 text_splitter_chunk_size: int = 1000,
                 text_splitter_chunk_overlap: int = 200,
                 debug: bool = False) -> None:
        self._debug = debug
        self._oembed = OllamaEmbeddings(base_url=ollama_base_url, model=ollama_model)
        self._vectorstore = Chroma(chroma_db_name, self._oembed, chroma_db_path)
        self._text_splitter_chunk_size = text_splitter_chunk_size
        self._text_splitter_chunk_overlap = text_splitter_chunk_overlap

    @property
    def vectorstore(self):
        return self._vectorstore

    def find_documents_by_source(self, source_path: str) -> List[Document]:
        results = self.vectorstore.search(
            query="dummy",
            search_type="similarity",
            filter={"source": source_path}
        )
        return results

    def get_list_of_stored_files(self) -> List[str]:
        all_documents = self.vectorstore._collection.get()
        if not all_documents['metadatas']:
            return []
        return list(set([m["source"] for m in all_documents['metadatas']]))

    def get_checksum_for_file(self, source_path: str) -> Optional[str]:
        docs = self.find_documents_by_source(source_path)
        if docs:
            return docs[0].metadata.get("checksum")
        return None

    def delete_by_source(self, source_path: str) -> None:
        if self._debug:
            print(f"  Deleting documents for {source_path}")
        self.vectorstore._collection.delete(where={"source": source_path})

    def should_reprocess(self, source_path: str, checksum: str, reload: bool) -> bool:
        docs = self.find_documents_by_source(source_path)
        if len(docs) == 0:
            return True
        if reload and any(doc.metadata.get("checksum") != checksum for doc in docs):
            return True
        return False

    def store_image_description(self, description: str, metadata: Dict[str, Any]) -> List[str]:
        doc = Document(page_content=description, metadata=metadata)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._text_splitter_chunk_size,
            chunk_overlap=self._text_splitter_chunk_overlap
        )
        splits = text_splitter.split_documents([doc])

        if not splits:
            return []

        if self._debug:
            print(f"  Storing {len(splits)} chunk(s) for {metadata.get('source', 'unknown')}")

        return self._vectorstore.add_documents(splits)

    def search_photos(self, query: str, limit: int = 10,
                      filter_dict: dict = None) -> List[Document]:
        search_kwargs = {"k": limit}
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        return self.vectorstore.similarity_search(query, **search_kwargs)

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        all_docs = self.vectorstore._collection.get(include=["metadatas", "documents"])
        results = []
        if all_docs['metadatas']:
            for i, meta in enumerate(all_docs['metadatas']):
                entry = dict(meta)
                if all_docs['documents'] and i < len(all_docs['documents']):
                    entry['description'] = all_docs['documents'][i]
                results.append(entry)
        return results
