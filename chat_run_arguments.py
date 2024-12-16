from argparse import ArgumentParser, Namespace

# interface for the run arguments which we will return from the parse method
class ChatRunArguments:
    def __init__(self, namespace: Namespace):
        self._system_prompt = namespace.system_prompt
        self._chroma_db_name = namespace.chroma_db_name
        self._chroma_db_path = namespace.chroma_db_path

    @property
    def system_prompt(self):
        return self._system_prompt
    
    @property
    def chroma_db_name(self):
        return self._chroma_db_name
    
    @property
    def chroma_db_path(self):
        return self._chroma_db_path


class RunArguments:
    def __init__(self):
        self.parser = ArgumentParser(description='Run the program')

        self.parser.add_argument(
            '--system-prompt',
            type=str,
            required=False,
            default=None,
            help='System prompt')

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
        
    def parse(self) -> ChatRunArguments:
        return_namespace = self.parser.parse_args()
        return ChatRunArguments(return_namespace)
        