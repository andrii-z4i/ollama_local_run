from argparse import ArgumentParser, Namespace
from typing import Sequence


class AnalyzePhotosRunArguments:
    """Encapsulates the arguments for the photo analysis pipeline."""

    def __init__(self, namespace: Namespace):
        self._directory_to_analyze = namespace.directory_to_analyze
        self._extensions = namespace.extensions
        self._reload = namespace.reload
        self._verbose = namespace.verbose
        self._chroma_db_name = namespace.chroma_db_name
        self._chroma_db_path = namespace.chroma_db_path
        self._exclude_subdirectories = namespace.exclude_subdirectories
        self._vision_model = namespace.vision_model
        self._ollama_url = namespace.ollama_url
        self._geocode_cache = namespace.geocode_cache

    @property
    def directory_to_analyze(self):
        return self._directory_to_analyze

    @property
    def extensions(self):
        return self._extensions

    @property
    def reload(self):
        return self._reload

    @property
    def verbose(self):
        return self._verbose

    @property
    def chroma_db_name(self):
        return self._chroma_db_name

    @property
    def chroma_db_path(self):
        return self._chroma_db_path

    @property
    def exclude_subdirectories(self):
        return self._exclude_subdirectories

    @property
    def vision_model(self):
        return self._vision_model

    @property
    def ollama_url(self):
        return self._ollama_url

    @property
    def geocode_cache(self):
        return self._geocode_cache


class RunArguments:
    """Parses command-line arguments for the photo analysis pipeline."""

    def __init__(self):
        self.parser = ArgumentParser(
            description='Analyze photos with a vision model and store descriptions in ChromaDB')

        self.parser.add_argument(
            '--directory-to-analyze',
            type=str,
            required=True,
            help='Directory containing photos to analyze')

        self.parser.add_argument(
            '--extensions',
            nargs='+',
            required=False,
            default=['jpg', 'jpeg', 'png', 'heic', 'webp'],
            help='Image extensions to process (default: jpg jpeg png heic webp)')

        self.parser.add_argument(
            '--reload',
            action='store_true',
            required=False,
            default=False,
            help='Re-analyze photos whose checksum has changed')

        self.parser.add_argument(
            '--verbose',
            action='store_true',
            required=False,
            default=False,
            help='Enable verbose output')

        self.parser.add_argument(
            '--chroma-db-name',
            type=str,
            required=False,
            default='photo_db',
            help='ChromaDB collection name (default: photo_db)')

        self.parser.add_argument(
            '--chroma-db-path',
            type=str,
            required=False,
            default='./chroma_db',
            help='Path to ChromaDB storage (default: ./chroma_db)')

        self.parser.add_argument(
            '--exclude-subdirectories',
            type=str,
            nargs='+',
            required=False,
            default=[],
            help='Subdirectory names to skip')

        self.parser.add_argument(
            '--vision-model',
            type=str,
            required=False,
            default='gemma3',
            help='Ollama vision model to use (default: gemma3)')

        self.parser.add_argument(
            '--ollama-url',
            type=str,
            required=False,
            default='http://localhost:11434',
            help='Ollama API base URL (default: http://localhost:11434)')

        self.parser.add_argument(
            '--geocode-cache',
            type=str,
            required=False,
            default='.geocode_cache.json',
            help='Path to geocoding cache file (default: .geocode_cache.json)')

    def parse(self, args: Sequence[str] = None) -> AnalyzePhotosRunArguments:
        return_namespace = self.parser.parse_args(args=args)
        return AnalyzePhotosRunArguments(return_namespace)
