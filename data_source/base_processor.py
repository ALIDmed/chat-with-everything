from abc import ABC
from langchain.text_splitter import CharacterTextSplitter

class BaseProcessor(ABC):

    def __init__(self, chunk_size=1000, chunk_overlap=100, text_splitter=None):
        self.text_splitter = text_splitter or CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def load(self):
        """
        load data
        """

    def chunk(self):
        """
        data chunking
        """

    def process(self):
        """
        process data
        """