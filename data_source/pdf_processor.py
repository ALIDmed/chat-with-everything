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