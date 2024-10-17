from .base_processor import BaseProcessor
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter

class ArticleProcessor(BaseProcessor):
    def __init__(self, url, text_splitter=None):
        super().__init__(text_splitter=text_splitter)
        self.url = url

    def load(self):
        loader = WebBaseLoader(self.url)
        documents = loader.load()
        self.documents = documents

    def chunk(self):
        self.docs = self.text_splitter.split_documents(self.documents)

    def process(self):
        self.load()
        self.chunk()
        return self.docs