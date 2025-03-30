from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from fastapi import FastAPI

import os
load_dotenv()

app = FastAPI()


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))


def multiply(a: int, b: int) -> float:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


def add(a: int, b: int) -> float:
    """add a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


def subtract(a: int, b: int) -> float:
    """subtract a and b.

    Args:
        a: first int
        b: second int
    """
    return a - b


all_tools = [multiply, add, subtract]
llm_with_tools = llm.bind_tools(all_tools)


def assistant(state: MessagesState) -> MessagesState:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Build graph
builder: StateGraph = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(all_tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

memory: MemorySaver = MemorySaver()
graph: CompiledStateGraph = builder.compile(checkpointer=memory)

# messages = [HumanMessage(
#     content="What is a addition of 2 and 3 and  3 multiply 4 and 4 multiply 5")]
config1 = {"configurable": {"thread_id": "1"}}
# responses = graph.invoke({"messages": messages}, config1)

# for response in responses["messages"]:
#     response.pretty_print()


@app.get("/{chat}")
def read_root(chat: str):
    messages = [HumanMessage(
        content=chat)]
    responses = graph.invoke({"messages": messages}, config1)
    for response in responses["messages"]:
        response.pretty_print()
    return responses
