from data_source.youtube_processor import YoutubeProcessor
from data_source.pdf_processor import PdfProcessor
from data_source.article_processor import ArticleProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embeddings.embeddings import GeminiEmbeddings

# Initialize the processors
pdf_processor = PdfProcessor(file_path="./files/Ali_AMZYL_Resume.pdf")

docs = pdf_processor.process()

embeddings = GeminiEmbeddings(db_dir='./db')
embeddings.generate_embdeddings(docs)

# Retrieve relevant documents
query = "what is movie whisper"
relevant_docs = embeddings.retrieve(query)

print(relevant_docs)