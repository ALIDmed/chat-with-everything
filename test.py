from data_source.youtube_processor import YoutubeProcessor
from data_source.pdf_processor import PdfProcessor
from data_source.article_processor import ArticleProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embeddings.embeddings import GeminiEmbeddings

# Initialize the processors
pdf_processor = YoutubeProcessor(
    file_paths=["./files/Ali_AMZYL_Resume.pdf", "./files/ds_interview_prep.pdf"]
    )

docs = pdf_processor.process()
for doc in docs:
    print(doc.page_content)
    print('-'*50)
    print("\n\n")

print(pdf_processor.vector_store_name)
# embeddings = GeminiEmbeddings(db_dir='./db')
# embeddings.generate_embdeddings(docs)

# # Retrieve relevant documents
# query = "what is movie whisper"
# relevant_docs = embeddings.retrieve(query)

# print(relevant_docs)