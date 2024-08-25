from embeding import Embedding
from langchain_community.llms import Ollama
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain


class OllamaLLM:
    def __init__(self, 
                 embedding: Embedding,
                 model_name='gemma2:2b'):
        self.model = Ollama(
            base_url='http://localhost:11434',
            model=model_name
        )
        self.embedding = embedding
    
    def talk(self, human_prompt):
        if self.embedding.vectorstore is None:
            return "Nothing to talk about. Please load some documents first."
        
        retriever = self.embedding.vectorstore.as_retriever()
        # 2. Incorporate the retriever into a question-answering chain.
        system_prompt = (
            "You are an assistant for Traffic Management Topology definitions. "
            "Use the following pieces of retrieved context to answer "
            "the question. The information you have access to represents the "
            "topology definitions per policy. servicedef.yaml is the Environment, "
            "top level domain, weighted switch definition. While other files are "
            "representing policies and their definitions. "
            " If you don't know the answer, say that you "
            "don't know. Keep the answer concise."
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

    

