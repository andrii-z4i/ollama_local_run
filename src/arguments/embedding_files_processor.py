from argparse import ArgumentParser, Namespace
from typing import Sequence

# interface for the run arguments which we will return from the parse method
class LlmRunArguments:
    """
    LlmRunArguments is a class that encapsulates the arguments required for running an LLM (Large Language Model) process.
    Attributes:
        directory_to_analyze (str): The directory to analyze.
        extensions (list): List of file extensions to consider.
        reload (bool): Flag indicating whether to reload the data.
        verbose (bool): Flag indicating whether to run in verbose mode.
        chroma_db_name (str): The name of the Chroma database.
        chroma_db_path (str): The path to the Chroma database.
        exclude_subdirectories (bool): Flag indicating whether to exclude subdirectories.
    Args:
        namespace (Namespace): A namespace object containing the arguments.
    """
    def __init__(self, namespace: Namespace):
        self._directory_to_analyze = namespace.directory_to_analyze
        self._extensions = namespace.extensions
        self._reload = namespace.reload
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



class RunArguments:
    """
    RunArguments is a class that encapsulates the command-line arguments for running the program.
    Attributes:
        parser (ArgumentParser): The argument parser instance.
    Methods:
        __init__():
            Initializes the RunArguments instance and sets up the command-line arguments.
        parse(args: Sequence[str] = None) -> LlmRunArguments:
            Parses the command-line arguments and returns an LlmRunArguments instance.
    Command-line Arguments:
        --directory-to-analyze (str, required):
            Directory to analyze.
        --extensions (list of str, required):
            Extensions to analyze.
        --reload (bool, optional, default=False):
            Reload embeddings if checksum is different.
        --verbose (bool, optional, default=False):
            Enable verbose mode.
        --chroma-db-name (str, optional, default='chroma_db'):
            Chroma database name.
        --chroma-db-path (str, optional, default='./chroma_db'):
            Path to the chroma database.
        --exclude-subdirectories (list of str, optional, default=[]):
            Subfolders to exclude from analysis.
    """
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
        
        # argument for the soft reload of embeddings (it will reload only if file is not in the vectorstore or is different by content)
        self.parser.add_argument(
            '--reload', 
            action='store_true',
            required=False,
            default=False,
            help='Reload embeddings if checksum is different')

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
        