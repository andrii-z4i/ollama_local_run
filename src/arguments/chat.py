from argparse import ArgumentParser, Namespace

# interface for the run arguments which we will return from the parse method
class ChatRunArguments:
    """
    ChatRunArguments is a class that encapsulates the arguments required for running a chat session.
    Attributes:
        system_prompt (str): The system prompt to be used in the chat session.
        chroma_db_name (str): The name of the Chroma database.
        chroma_db_path (str): The file path to the Chroma database.
    Args:
        namespace (Namespace): A namespace object containing the arguments for the chat session.
    """
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
    """
    A class to handle the parsing of command-line arguments for running the program.
    Attributes:
    parser : ArgumentParser
        An ArgumentParser object to handle the command-line arguments.
    
    Methods:
    __init__():
        Initializes the RunArguments class and sets up the argument parser with the required arguments.
    parse() -> ChatRunArguments:
        Parses the command-line arguments and returns a ChatRunArguments object containing the parsed values.
    """
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
        