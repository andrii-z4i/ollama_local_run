import hashlib
import os
from typing import Iterable, List, Tuple
from embeding import Embedding, EmbeddingProcessMode
import asyncio


class FilesProcessor:
    def __init__(self, 
                 embedding: Embedding, 
                 directory: str, 
                 extensions: List[str], 
                 exclude_subfolders: List[str],
                 process_mode: EmbeddingProcessMode) -> None:
        self.embedding = embedding
        self.directory = directory
        self.extensions = extensions
        self.exclude_subfolders = exclude_subfolders
        self.process_mode = process_mode
    
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

    async def process_files(self):
        tasks = []
        for file_for_processing in self.enumerate_files(self.directory):
            tasks.append(self.embedding.aload_content_from_path(file_for_processing[0], file_for_processing[1], self.process_mode))
        
        await asyncio.gather(*tasks)
        