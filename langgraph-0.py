from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from fastapi import FastAPI
import os
load_dotenv()
app=FastAPI()
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
# Create a message
# msg = HumanMessage(content="What is LangChain", name="Lance")

# # Message list
# messages = [msg]

# # Invoke the model with a list of messages

# response = llm.invoke(messages)
# print(response)

tavily_search = TavilySearchResults(max_results=3 ,api_key=os.getenv("TAVILY_API_KEY"))

search_docs = tavily_search.invoke("What is  score of pak vs NZ match live ?")

print(search_docs)


@app.get("/")
async def chat():
    return search_docs
    