from data_source.youtube_processor import YoutubeProcessor
from data_source.pdf_processor import PdfProcessor
from data_source.article_processor import ArticleProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter


url = "https://medium.com/@tejpal.abhyuday/retrieval-augmented-generation-rag-from-basics-to-advanced-a2b068fd576c"
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
processor = ArticleProcessor(url=url, text_splitter=splitter)
docs = processor.process()
for doc in docs:
    print(doc.page_content)
    print("\n\n\n\n")