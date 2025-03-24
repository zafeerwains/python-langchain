from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.graph import MessagesState

import random
import os
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))


def multiply(a: int, b: int) -> float:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    # return a / b
    # return a / b
    return a + b


llm_with_tools = llm.bind_tools([multiply])


def tool_calling_llm(state: MessagesState) -> MessagesState:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Build graph
builder: StateGraph = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)
graph: CompiledStateGraph = builder.compile()

messages = [HumanMessage(content="What is a product of 2 and 3")]
responses = graph.invoke({"messages": messages})

for response in responses["messages"]:
    print(response)
