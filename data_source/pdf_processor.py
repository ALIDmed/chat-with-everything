from .base_processor import BaseProcessor
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter

class PdfProcessor(BaseProcessor):
    def __init__(self, file_path, text_splitter=None):
        super().__init__(text_splitter=text_splitter)
        self.file_path = file_path

    def load(self):
        loader = PyMuPDFLoader(self.file_path)
        documents = loader.load()
        self.documents = documents

    def chunk(self):
        self.docs = self.text_splitter.split_documents(self.documents)

    def process(self):
        self.load()
        self.chunk()
        return self.docs