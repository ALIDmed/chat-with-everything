import os
from data_source.pdf_processor import PdfProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embeddings.embeddings import GeminiEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from chatwrapper import ChatWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
INPUT_SOURCE = ["./files/ds_interview_prep.pdf"]
DB_DIR = "./db"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
processor = PdfProcessor(file_paths=INPUT_SOURCE, text_splitter=text_splitter)
docs = processor.process()
vectore_store_name = processor.generate_vector_store_name()

embeddings = GeminiEmbeddings(db_dir=os.path.join(DB_DIR, vectore_store_name))
embeddings.generate_embdeddings(docs)

retriever = embeddings.setup_retiever()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.getenv('GOOGLE_API_KEY'))

chat_wrapper = ChatWrapper(embeddings, llm)
chat_wrapper.setup_retrieval_chain()

query = "What is the first convolutional network"
# res = chat_wrapper.chat(query=query, chat_history=[])
# for chunk in res:
#     print(chunk, end="", flush=True)

print("Start chating with the AI. type 'exit' to end the conversation")
chat_history = []
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    res = chat_wrapper.chat(query=query, chat_history=[])
    full_response = ""
    print("AI: ", end="")
    for chunk in res:
        full_response += chunk
        print(chunk, end="", flush=True)

    chat_history.append(HumanMessage(query))
    chat_history.append(AIMessage(full_response))