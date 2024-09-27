from embeding import Embedding
from langchain_community.llms import Ollama
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain


class OllamaLLM:
    def __init__(self, 
                 embedding: Embedding,
                 model_name='gemma2:2b',
                 system_prompt=None):
        self.model = Ollama(
            base_url='http://localhost:11434',
            model=model_name
        )
        self.embedding = embedding
        self.system_prompt = system_prompt
    
    def talk(self, human_prompt):
        if self.embedding.vectorstore is None:
            return "Nothing to talk about. Please load some documents first."
        
        retriever = self.embedding.vectorstore.as_retriever()
        # 2. Incorporate the retriever into a question-answering chain.
        system_prompt = (
            f"{self.system_prompt}"
            "\n\n"
            "{context}"
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

    

