from abc import ABC, abstractmethod
from langchain.text_splitter import CharacterTextSplitter
import uuid

class BaseProcessor(ABC):

    def __init__(self, chunk_size=200, chunk_overlap=20, text_splitter=None):
        self.text_splitter = text_splitter or CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    @abstractmethod
    def load(self):
        """
        load data
        """
    def chunk(self):
        """
        data chunking
        """
        self.docs = self.text_splitter.split_documents(self.documents)

    def generate_vector_store_name(self):
            if not self.documents:
                raise Exception("call process() before generating vector store name")
            
            combined_content = "".join([doc.page_content for doc in self.documents])
            self.vector_store_name = str(uuid.uuid5(uuid.NAMESPACE_DNS, combined_content))
            return self.vector_store_name

    def process(self):
        """
        process data
        """
        self.load()
        self.chunk()
        self.generate_vector_store_name()
        return self.docs