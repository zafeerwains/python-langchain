from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph  # type
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState
from fastapi import FastAPI

import random
import os
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers.
    args:
        a: int
        b: int
    returns:
        int
    """
    return a / b
llm_with_tools = llm.bind_tools([multiply])

# response = llm_with_tools.invoke( [HumanMessage(content=f"Deposit Money in Ahmad Account. His acc number is 00123", name="Muhammad")])
# print(response)

# class LlmWithToolSState(TypedDict):
#     messages: Annotated[list, add_messages]
    
class LlmWithToolSState(MessagesState):
    pass

# 1. tool calling llm ki Node

def call_llm(state: LlmWithToolSState):
  messages = state["messages"]
  call_response = llm_with_tools.invoke(messages)
  # messages.append(call_response)

  # return {"messages": messages}
  return {"messages": [call_response]}
# 2. Graph


builder: StateGraph = StateGraph(LlmWithToolSState)

# define nodes
builder.add_node("call_llm_with_tools", call_llm)

# define edges
builder.add_edge(START, "call_llm_with_tools")
builder.add_edge("call_llm_with_tools", END)

# build graph
graph :CompiledStateGraph = builder.compile()
# response=graph.invoke({"messages": [HumanMessage(content="plus 10 by 2", name="Muhammad")]})
# print(response)

app = FastAPI()

@app.get("/chat/{query}")
def get_content(query: str):
    try:
        # result = graph.invoke({"messages": [HumanMessage("content", query)]})
        response=graph.invoke({"messages": [HumanMessage(content=query, name="Muhammad")]})
        return response
    except Exception as e:
        return {"output": str(e)}