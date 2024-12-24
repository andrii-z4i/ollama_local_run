from src.embeding_manager import EmbeddingManager
from langchain_ollama import OllamaLLM as _OllamaLLM
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


class OllamaLLM:
    """
    OllamaLLM is a class that integrates a language model with a vector store for retrieval-augmented generation (RAG).
    Attributes:
        model (object): The language model instance.
        system_prompt (str): The system prompt to be used in the conversation.
        _oembed (OllamaEmbeddings): The embeddings model for vector store.
        _vectorstore (Chroma): The vector store instance.
    Args:
        model_name (str): The name of the language model to use. Default is "gemma2:2b".
        ollama_base_url (str): The base URL for the Ollama API. Default is "http://localhost:11434".
        ollama_model (str): The model to use for embeddings. Default is "nomic-embed-text".
        chroma_db_name (str): The name of the Chroma database. Default is "chroma_db".
        chroma_db_path (str): The path to the Chroma database. Default is "./chroma_db".
        system_prompt (str): The system prompt to be used in the conversation. Default is None.
    Methods:
        talk(human_prompt):
            Generates a response to the given human prompt using the language model and vector store.
            Args:
                human_prompt (str): The prompt provided by the user.
            Returns:
                str: The generated response based on the context retrieved from the vector store.
    """
    def __init__(self, 
                 model_name:      str = "gemma2:2b",
                 ollama_base_url: str = "http://localhost:11434",
                 ollama_model:    str = "nomic-embed-text", 
                 chroma_db_name:  str = "chroma_db",
                 chroma_db_path:  str = "./chroma_db",
                 system_prompt:   str = None):
        self.model = _OllamaLLM(
            base_url='http://localhost:11434',
            model=model_name
        )
        self.system_prompt = system_prompt
        self._oembed = OllamaEmbeddings(base_url=ollama_base_url, model=ollama_model)
        self._vectorstore = Chroma(chroma_db_name, self._oembed, chroma_db_path)
    
    def talk(self, human_prompt):
        """
        Engage in a conversation based on the provided human prompt.
        This method uses a vector store to retrieve relevant documents and incorporates them into a 
        question-answering chain to generate a response. If no documents are loaded in the vector store, 
        it returns a message indicating that there is nothing to talk about.
        Args:
            human_prompt (str): The prompt or question provided by the user.
        Returns:
            str: The generated response based on the context from the retrieved documents or a message 
                 indicating that the information is not available in the provided context.
        """
        if self._vectorstore is None:
            return "Nothing to talk about. Please load some documents first."
        
        retriever = self._vectorstore.as_retriever()
        # 2. Incorporate the retriever into a question-answering chain.
        system_prompt = (
            f"{self.system_prompt}"
            "\n\n"
            "{context}"
            "You must answer the user's question strictly based on the context provided. "
            "If the answer cannot be determined from the context, respond with 'The information is not available in the provided context.'"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.model, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        result = rag_chain.invoke({"input": human_prompt})
        return result["answer"]

    

