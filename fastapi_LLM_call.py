import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str


load_dotenv()
app = FastAPI()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    google_api_key=os.getenv('GOOGLE_API_KEY')
)


def get_response(query: str) -> str:
    response = llm.invoke(query)
    return response


@app.post("/chat")
async def chat(request: ChatRequest):
    response = llm.invoke(request.query)
    # response = get_response(request.query)
    return {response.content}
