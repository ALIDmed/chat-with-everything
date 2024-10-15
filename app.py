from data_source.pdf_processor import PdfProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embeddings.embeddings import Embeddings

pdf_file_path = "./files/Ali_AMZYL_Resume.pdf"
pdf = PdfProcessor(
    file_path=pdf_file_path,
    text_splitter=RecursiveCharacterTextSplitter()
)

pdf.load()
pdf.chunk()

# TODO: change persistent_dir to db_dir and add unique name to each file
embeddings = Embeddings(persistent_dir="./db")
embeddings.create_vectore_store(pdf.docs)