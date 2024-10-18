import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

class EmbeddingStrategy(ABC):
    def _initialize_db(self) :
        pass

class GeminiEmbeddings(EmbeddingStrategy):
    def __init__(self, db_dir, embedding_model="models/embedding-001"):
        self.embedding_model = embedding_model
        self.db_dir = db_dir
        self.retriever = None
        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            model=self.embedding_model
        )

    def generate_embdeddings(self, docs):

        self.db = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )

    def retrieve(self, query, search_type="similarity",search_kwargs={'k':3}):
        if self.retriever is None:
            self.retriever = self.db.as_retriever(
                search_type=search_type, 
                search_kwargs=search_kwargs
            )

        relevant_docs = self.retriever.invoke(query)
        return relevant_docs