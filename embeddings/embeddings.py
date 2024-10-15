import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()
class Embeddings:
    def __init__(self, persistent_dir, model="models/embedding-001"):
        self.persistent_dir = os.path.abspath(persistent_dir)
        self.embeddings = GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=os.getenv('GOOGLE_API_KEY')
            )
        self.db = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persistent_dir
        )
    
    def create_vectore_store(self, docs):
        self.db.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persistent_dir
        )