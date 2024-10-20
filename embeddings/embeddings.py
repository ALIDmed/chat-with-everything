import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from langchain_chroma.vectorstores import Chroma
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
        self.db = None

    def generate_embdeddings(self, docs):
        if os.path.exists(self.db_dir):
            print("Vector store already exists")
            self.setup_db()
            return
        print("Generating vector store...")
        self.db = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )
    
    def setup_db(self):
        if not self.db:
            self.db = Chroma(embedding_function=self.embeddings, persist_directory=self.db_dir)
        return self.db

    def setup_retiever(self, search_type="similarity",search_kwargs={'k':3}):
        self.setup_db()
        self.retriever = self.db.as_retriever(
            search_type=search_type, 
            search_kwargs=search_kwargs
        )
        return self.retriever

    def retrieve(self, query):
        if self.retriever is None:
            self.setup_retiever()

        relevant_docs = self.retriever.invoke(query)
        return relevant_docs