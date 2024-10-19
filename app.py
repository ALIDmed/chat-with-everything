import os
from data_source.pdf_processor import PdfProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embeddings.embeddings import GeminiEmbeddings

INPUT_SOURCE = ["./files/ds_interview_prep.pdf"]
DB_DIR = "./db"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
processor = PdfProcessor(file_paths=INPUT_SOURCE, text_splitter=text_splitter)
docs = processor.process()
vectore_store_name = processor.generate_vector_store_name()

print(vectore_store_name)

embeddings = GeminiEmbeddings(db_dir=os.path.join(DB_DIR, vectore_store_name))
embeddings.generate_embdeddings(docs)

query = "What is upsampling and downsampling with examples?"
relevant_docs = embeddings.retrieve(query)

for relevant_doc in relevant_docs:
    print(relevant_doc.page_content)