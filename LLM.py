from embeding import Embedding
from langchain_ollama import OllamaLLM as _OllamaLLM
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


class OllamaLLM:
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

    

