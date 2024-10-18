import hashlib
from .base_processor import BaseProcessor
from langchain_community.document_loaders import YoutubeLoader

class YoutubeProcessor(BaseProcessor):
    def __init__(self, urls, text_splitter=None):
        super().__init__(text_splitter=text_splitter)
        self.urls = urls if isinstance(urls, list) else [urls]
        self.documents = []

    def load(self):
        for url in self.urls:
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
            documents = loader.load()
            self.documents.extend(documents)
        return documents