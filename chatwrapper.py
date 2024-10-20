from chat_templates import qa_system_prompt, contextualize_q_system_prompt
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class ChatWrapper:
    def __init__(self, embeddings, llm):
        self.embeddings = embeddings
        self.llm = llm
    
    def _setup_history_aware_retriever(self):
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        self.history_aware_retriever = create_history_aware_retriever(self.llm, self.embeddings.retriever, contextualize_q_prompt)

    def setup_retrieval_chain(self):
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self._setup_history_aware_retriever()
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, question_answer_chain)

    def chat(self, query, chat_history):
        response = self.rag_chain.stream({"input": query, "chat_history": chat_history})
        for chunk in response:
            if chunk.get('answer', None):
                yield chunk['answer']