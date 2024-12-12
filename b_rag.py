from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
import os
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()


llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

memory = ConversationBufferWindowMemory(k=5)

try:
    pdf_files = ["1.pdf", "2.pdf", "3.pdf", "4.pdf", "5.pdf", "6.pdf", "7.pdf", "8.pdf"]
    # pdf_files = ["9.pdf"]
    loaders = [PyPDFLoader(file_path) for file_path in pdf_files]

    # Load documents from each loader
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
except Exception as e:  
    print("Error while loading file:", e)

# Create embeddings
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Use a smaller chunk size to manage token limits
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Create the index with the specified embedding model and text splitter
index_creator = VectorstoreIndexCreator(
    embedding=embedding,
    text_splitter=text_splitter
)
index = index_creator.from_documents(docs)

# Query the index with the LLM
while True:
    human_message = input("How i can help you today? ")
    response = index.query(human_message, llm=llm, memory=memory)
    print(response)
