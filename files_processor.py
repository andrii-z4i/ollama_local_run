import hashlib
import os
from typing import Iterable, List, Tuple
from embeding import Embedding


class FilesProcessor:
    def __init__(self, 
                 embedding: Embedding, 
                 directory: str, 
                 extensions: List[str], 
                 exclude_subfolders: List[str],
                 reload: bool = False,
                 verbose: bool = False) -> None:
        self.embedding = embedding
        self.directory = directory
        self.extensions = extensions
        self.exclude_subfolders = exclude_subfolders
        self.reload = reload
        self.verbose = verbose
    
    def _calculate_checksum(self, file_path: str) -> str:
        # Calculate the checksum of the file by using hash algorithms
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        

    def enumerate_files(self, directory: str) -> Iterable[Tuple[str, str]]:
        for filename in os.listdir(directory):
            if filename in self.exclude_subfolders:
                continue
            full_path = os.path.join(directory, filename)
            # if filename ends with one of the extensions
            if any([filename.endswith(ext) for ext in self.extensions]):
                yield [full_path, self._calculate_checksum(full_path)]
            elif os.path.isdir(full_path):
                yield from self.enumerate_files(full_path)

    def process_files(self):
        print(f"Files will be processed in the reload mode: {self.reload}") if self.verbose else None
        _files_to_process = [file for file in self.enumerate_files(self.directory)]
        for file_for_processing in _files_to_process:
            self.embedding.load_content_from_path(file_for_processing[0], file_for_processing[1], self.reload)
        
        print(f"Files processed: {len(_files_to_process)}") if self.verbose else None
        # delete files from embedding that are not in the directory
        _files = self.embedding.get_list_of_stored_files()
        _files_to_compare_to = [file[0] for file in _files_to_process]
        
        for file in _files:
            if file not in _files_to_compare_to:
                self.embedding._delete_documents_by_path(file)
                print(f"File {file} was deleted from the vectorstore") if self.verbose else None
        
        
        