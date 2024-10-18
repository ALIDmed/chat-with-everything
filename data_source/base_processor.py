from abc import ABC, abstractmethod
from langchain.text_splitter import CharacterTextSplitter

class BaseProcessor(ABC):

    def __init__(self, chunk_size=200, chunk_overlap=20, text_splitter=None):
        self.text_splitter = text_splitter or CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    @abstractmethod
    def load(self):
        """
        load data
        """
    @abstractmethod
    def chunk(self):
        """
        data chunking
        """
    @abstractmethod
    def generate_vector_store_name(self):
        """
        generate vector store name
        """

    def process(self):
        """
        process data
        """
        self.load()
        self.chunk()
        self.generate_vector_store_name()
        return self.docs