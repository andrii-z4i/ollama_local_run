from argparse import ArgumentParser, Namespace

# interface for the run arguments which we will return from the parse method
class LlmRunArguments:
    def __init__(self, namespace: Namespace):
        self._directory_to_analyze = namespace.directory_to_analyze
        self._extensions = namespace.extensions
        self._force_reload = namespace.force_reload
        self._soft_reload = namespace.soft_reload
        self._verbose = namespace.verbose

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
        
        # argument for the force reload of embeddings
        self.parser.add_argument(
            '--force-reload', 
            action='store_true', 
            required=False,
            default=False,
            help='Force reload of embeddings')
        
        # argument for the soft reload of embeddings (it will reload only if file is not in the vectorstore)
        self.parser.add_argument(
            '--soft-reload', 
            action='store_true',
            required=False,
            default=False,
            help='Soft reload of embeddings')

        # argument for the verbose mode        
        self.parser.add_argument(
            '--verbose', 
            action='store_true', 
            required=False,
            default=False,
            help='Verbose mode')

    def parse(self) -> LlmRunArguments:
        return_namespace = self.parser.parse_args()
        return LlmRunArguments(return_namespace)
        