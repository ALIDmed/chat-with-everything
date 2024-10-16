from .base_processor import BaseProcessor
from langchain_community.document_loaders import YoutubeLoader

class YoutubeProcessor(BaseProcessor):
    def __init__(self, url, text_splitter=None):
        super().__init__(text_splitter=text_splitter)
        self.url = url

    def load(self):
        loader = YoutubeLoader.from_youtube_url(self.url, add_video_info=True)
        documents = loader.load()
        self.documents = documents
        return documents
    
    def chunk(self):
        self.docs = self.text_splitter.split_documents(self.documents)

    def process(self):
        self.load()
        self.chunk()
        return self.docs