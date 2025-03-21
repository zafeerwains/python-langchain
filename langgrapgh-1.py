from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from fastapi import FastAPI
import os
load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))