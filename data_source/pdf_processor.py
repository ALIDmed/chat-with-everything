from .base_processor import BaseProcessor
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import hashlib

class PdfProcessor(BaseProcessor):
    def __init__(self, file_paths, text_splitter=None):
        super().__init__(text_splitter=text_splitter)
        self.file_paths = file_paths if isinstance(file_paths, list) else [file_paths]
        self.documents = []

    def load(self):
        for file_path in self.file_paths:
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            self.documents.extend(documents)
        return documents

    def chunk(self):
        self.docs = self.text_splitter.split_documents(self.documents)

    def generate_vector_store_name(self):
        if not self.documents:
            raise Exception("call process() before generating vector store name")
        
        combined_content = "".join([doc.page_content for doc in self.documents])
        self.vector_store_name = hashlib.sha256(
            combined_content.encode('utf-8')
            ).hexdigest()