from data_source.youtube_processor import YoutubeProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
processor = YoutubeProcessor(url=
                             "https://www.youtube.com/watch?v=75uBcITe0gU",
                             text_splitter=splitter
                             )
docs = processor.process()
print(docs)