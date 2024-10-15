from .base_processor import BaseProcessor
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter

class PdfProcessor(BaseProcessor):
    def __init__(self, file_path, text_splitter=None, chunk_size=1000, overlap=100):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.text_splitter = text_splitter or CharacterTextSplitter(chunk_size=chunk_size, overlap=overlap)

    def load(self):
        loader = PyMuPDFLoader(self.file_path)
        documents = loader.load()
        self.documents = documents

    def chunk(self):
        self.docs = self.text_splitter.split_documents(self.documents)