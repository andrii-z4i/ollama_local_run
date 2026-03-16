from argparse import ArgumentParser, Namespace
from typing import Sequence


class McpServerRunArguments:
    """Encapsulates the arguments for the MCP server."""

    def __init__(self, namespace: Namespace):
        self._chroma_db_name = namespace.chroma_db_name
        self._chroma_db_path = namespace.chroma_db_path
        self._ollama_url = namespace.ollama_url

    @property
    def chroma_db_name(self):
        return self._chroma_db_name

    @property
    def chroma_db_path(self):
        return self._chroma_db_path

    @property
    def ollama_url(self):
        return self._ollama_url


class RunArguments:
    """Parses command-line arguments for the MCP server."""

    def __init__(self):
        self.parser = ArgumentParser(
            description='Run the Photo Memory MCP server')

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
            '--ollama-url',
            type=str,
            required=False,
            default='http://localhost:11434',
            help='Ollama API base URL (default: http://localhost:11434)')

    def parse(self, args: Sequence[str] = None) -> McpServerRunArguments:
        return_namespace = self.parser.parse_args(args=args)
        return McpServerRunArguments(return_namespace)
