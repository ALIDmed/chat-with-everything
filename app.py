import os
from data_source.pdf_processor import PdfProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embeddings.embeddings import Embeddings

pdf_file_path = "./files/Ali_AMZYL_Resume.pdf"
file_name = os.path.basename(pdf_file_path)
pdf = PdfProcessor(
    file_path=pdf_file_path,
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0)
)

docs = pdf.process()

embeddings = Embeddings(db_dir="./db")
store_name = embeddings.generate_store_name(file_name)
# embeddings.create_vectore_store(docs)
relevant_docs = embeddings.query_vectore_store("python")
print(relevant_docs)