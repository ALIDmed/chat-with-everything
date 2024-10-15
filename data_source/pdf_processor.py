from base_processor import BaseProcessor
from langchain_community.document_loaders import PyMuPDFLoader

class PdfProcessor(BaseProcessor):
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        pass

    def chunk(self):
        return super().chunk()