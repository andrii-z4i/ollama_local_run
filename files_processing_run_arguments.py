from argparse import ArgumentParser, Namespace
from typing import Sequence

# interface for the run arguments which we will return from the parse method
class LlmRunArguments:
    def __init__(self, namespace: Namespace):
        self._directory_to_analyze = namespace.directory_to_analyze
        self._extensions = namespace.extensions
        self._force_reload = namespace.force_reload
        self._soft_reload = namespace.soft_reload
        self._verbose = namespace.verbose
        self._chroma_db_name = namespace.chroma_db_name
        self._chroma_db_path = namespace.chroma_db_path
        self._exclude_subdirectories = namespace.exclude_subdirectories

    @property
    def directory_to_analyze(self):
        return self._directory_to_analyze
    
    @property
    def extensions(self):
        return self._extensions
    
    @property
    def force_reload(self):
        return self._force_reload
    
    @property
    def soft_reload(self):
        return self._soft_reload
    
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



class RunArguments:
    def __init__(self):
        self.parser = ArgumentParser(description='Run the program')

        self.parser.add_argument(
            '--directory-to-analyze', 
            type=str, 
            required=True,
            help='Directory to analyze')
        
        # add the extensions argument which will be a list of strings
        self.parser.add_argument(
            '--extensions', 
            nargs='+', 
            required=True,
            help='Extensions to analyze')
        
        # argument for the force reload of embeddings (it will reload all embeddings)
        self.parser.add_argument(
            '--force-reload', 
            action='store_true', 
            required=False,
            default=False,
            help='Force reload of embeddings')
        
        # argument for the soft reload of embeddings (it will reload only if file is not in the vectorstore or is different by content)
        self.parser.add_argument(
            '--soft-reload', 
            action='store_true',
            required=False,
            default=True,
            help='Soft reload of embeddings')

        # argument for the verbose mode        
        self.parser.add_argument(
            '--verbose', 
            action='store_true', 
            required=False,
            default=False,
            help='Verbose mode')

        self.parser.add_argument(
            '--chroma-db-name',
            type=str,
            required=False,
            default='chroma_db',
            help='Chroma db name')
        
        self.parser.add_argument(
            '--chroma-db-path',
            type=str,
            required=False,
            default='./chroma_db',
            help='Chroma db path')
        
        self.parser.add_argument(
            '--exclude-subdirectories',
            type=str,
            nargs='+',
            required=False,
            default=[],
            help='Subfolders to exclude')


    def parse(self, args: Sequence[str] = None) -> LlmRunArguments:
        return_namespace = self.parser.parse_args(args=args)
        return LlmRunArguments(return_namespace)
        