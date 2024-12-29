from typing_extensions import TypedDict
from typing import Annotated, List, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from fastapi import FastAPI
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import os
load_dotenv()


# -------------------- Setup and Document Loading --------------------
# Initialize LLM and Embedding model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Text splitter
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Load documents
try:
    pdf_files = ["notesPDC.pdf"]
    loaders = [PyPDFLoader(file_path) for file_path in pdf_files]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
except Exception as e:
    print("Error while loading file:", e)
    docs = []  # Handle gracefully by creating an empty doc list

# Create index
index = None
if docs:  # Check if docs are loaded before creating index
    index_creator = VectorstoreIndexCreator(
    embedding=embedding,
    text_splitter=text_splitter)
    index = index_creator.from_documents(docs)



# -------------------- State Definition --------------------
class MessagesState(TypedDict):
    messages: Annotated[List[Dict], add_messages]


# -------------------- Agent Functions --------------------
def assistant(state: MessagesState):
    """Queries the document index with the latest user input."""
    global index
    if index is None:
        return {"messages": [AIMessage(content="No document index has been created")]}

    try:
        # Extract the content of the last user message
        user_message = state["messages"][-1].content
        response = index.query(user_message, llm=llm)  # Use the llm to query
        return {"messages": [AIMessage(content=response)]}
    except Exception as e:
        print("Error querying the index:", e)
        return {"messages": [AIMessage(content=f"Error querying the index: {str(e)}")]}


# -------------------- LangGraph Configuration --------------------
# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# -------------------- FastAPI Endpoint --------------------
app = FastAPI()

@app.get("/chat/{query}")
def get_content(query: str):
    try:
        config = {"configurable": {"thread_id": "1"}}
        result = graph.invoke({"messages": [HumanMessage(content=query)]}, config)
        return result
    except Exception as e:
        return {"output": str(e)}


