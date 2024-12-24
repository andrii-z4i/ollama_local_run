import hashlib
import os
from typing import Iterable, List, Tuple
from src.embeding_manager import EmbeddingManager


class FilesProcessor:
    """
    A class to process files in a specified directory, calculate their checksums, and manage their embeddings.
    Attributes:
        embedding (EmbeddingManager): An instance of EmbeddingManager to handle file embeddings.
        directory (str): The directory to process files from.
        extensions (List[str]): A list of file extensions to include in processing.
        exclude_subfolders (List[str]): A list of subfolders to exclude from processing.
        reload (bool): A flag to indicate if files should be reloaded. Default is False.
        verbose (bool): A flag to indicate if verbose output should be printed. Default is False.
    Methods:
        _calculate_checksum(file_path: str) -> str:
            Calculates the SHA-256 checksum of a file.
        enumerate_files(directory: str) -> Iterable[Tuple[str, str]]:
            Recursively enumerates files in the directory, yielding file paths and their checksums.
        process_files():
            Processes files in the directory, loads their content into the embedding manager, and deletes files from the embedding manager that are no longer in the directory.
    """
    def __init__(self, 
                 embedding: EmbeddingManager, 
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
        """
        Calculate the SHA-256 checksum of a file.

        Args:
            file_path (str): The path to the file for which the checksum is to be calculated.

        Returns:
            str: The SHA-256 checksum of the file as a hexadecimal string.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
        

    def _enumerate_files(self, directory: str) -> Iterable[Tuple[str, str]]:
        """
        Enumerates files in a given directory and its subdirectories, yielding file paths and their checksums.

        Args:
            directory (str): The directory to enumerate files from.

        Yields:
            Tuple[str, str]: A tuple containing the full file path and its checksum.

        Notes:
            - Files in subdirectories listed in `self.exclude_subfolders` are skipped.
            - Only files with extensions listed in `self.extensions` are processed.
        """
        for filename in os.listdir(directory):
            if filename in self.exclude_subfolders:
                continue
            full_path = os.path.join(directory, filename)
            # if filename ends with one of the extensions
            if any([filename.endswith(ext) for ext in self.extensions]):
                yield [full_path, self._calculate_checksum(full_path)]
            elif os.path.isdir(full_path):
                yield from self._enumerate_files(full_path)

    def process_files(self):
        """
        Processes files in the specified directory by loading their content into the embedding and 
        deleting files from the embedding that are no longer present in the directory.
        If the 'verbose' attribute is set to True, prints messages about the processing status.
        Steps:
        1. Prints a message indicating the start of file processing if verbose mode is enabled.
        2. Enumerates all files in the specified directory.
        3. Loads the content of each file into the embedding.
        4. Prints the number of processed files if verbose mode is enabled.
        5. Retrieves the list of files currently stored in the embedding.
        6. Compares the stored files with the files in the directory.
        7. Deletes files from the embedding that are no longer present in the directory.
        8. Prints a message for each file deleted from the embedding if verbose mode is enabled.
        Attributes:
            directory (str): The directory containing the files to be processed.
            reload (bool): A flag indicating whether to reload the files.
            verbose (bool): A flag indicating whether to print verbose messages.
            embedding (Embedding): An instance of the Embedding class used to load and manage file content.
        """
        print(f"Files will be processed in the reload mode: {self.reload}") if self.verbose else None
        _files_to_process = [file for file in self._enumerate_files(self.directory)]
        for file_for_processing in _files_to_process:
            self.embedding.load_content_from_path(file_for_processing[0], file_for_processing[1], self.reload)
        
        print(f"Files processed: {len(_files_to_process)}") if self.verbose else None
        # delete files from embedding that are not in the directory
        _files = self.embedding.get_list_of_stored_files()
        _files_to_compare_to = [file[0] for file in _files_to_process]
        
        for file in _files:
            if file not in _files_to_compare_to:
                self.embedding._delete_documents_by_path(file)
        
        
        