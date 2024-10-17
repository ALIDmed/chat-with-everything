import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

class EmbeddingStrategy(ABC):
    @abstractmethod
    def _initialize_db(self) :
        pass

class GeminiEmbeddings(EmbeddingStrategy):
    def __init__(self, db_dir, embedding_model="models/embedding-001"):
        self.embedding_model = embedding_model
        self.db_dir = db_dir
        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            model=self.embedding_model
        )

        # os.makedirs(self.db_dir, exist_ok=True)

    def generate_embdeddings(self, docs):

        Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )