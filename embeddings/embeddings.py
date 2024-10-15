import os
import hashlib
from abc import abstractmethod
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
class Embeddings:
    def __init__(self, db_dir, model="models/embedding-001"):
        self.db_dir = os.path.abspath(db_dir)
        self.embeddings = GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=os.getenv('GOOGLE_API_KEY')
            )
        self.db = None

    def generate_store_name(self, file_name):
        self.file_name = file_name
        file_name_hash = hashlib.md5(f"{file_name}".encode()).hexdigest()
        store_name = f"store_{file_name_hash}"
        self.persistent_dir = os.path.join(self.db_dir, store_name)
        return self.persistent_dir
    
    def get_db(self):
        if not self.db:
            self.db = Chroma(embedding_function=self.embeddings, persist_directory=self.persistent_dir)
        return self.db

    def create_vectore_store(self, docs):
        if os.path.exists(self.persistent_dir):
            raise Exception(f"Store {self.persistent_dir} already exists")
        
        self.db = self.get_db()
        self.db.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persistent_dir
        )

    def query_vectore_store(self, query):
        if not os.path.exists(self.persistent_dir):
            raise Exception(f"The store {self.persistent_dir} does not exist")
        
        self.db = self.get_db()
        retriever = self.db.as_retriever(search_type='similarity', search_kwargs={'k': 3})
        relevant_docs = retriever.invoke(query)
        return relevant_docs
        