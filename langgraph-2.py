from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph  # type
from dotenv import load_dotenv
from IPython.display import Image, display
import random
import os
load_dotenv()


class Conditional_edge(TypedDict):
    user_input: str


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))


def node_1(state: Conditional_edge) -> Conditional_edge:
    print("---Node 1 State---", state)
    ai_msg = llm.invoke(state['user_input'])
    # print("AI_MSG", ai_msg)
    return {"user_input": ai_msg.content}


def node_2(state: Conditional_edge) -> Conditional_edge:
    print("---Node 2 State---", state)
    ai_msg = llm.invoke(
        state['user_input']+"I want to know that anyperson from GCUF work on it")
    return {"user_input": ai_msg.content}


def node_3(state: Conditional_edge) -> Conditional_edge:
    print("---Node 3 State---", state)
    ai_msg = llm.invoke(
        state['user_input']+"I want to know that anyperson from UAF Pakistan work on it")
    return {"user_input": ai_msg.content}


def Conditional_node(state: Conditional_edge) :
    number = random.random()
    if number > 0.5:
        return "node_2"
    return "node_3"


builder: StateGraph = StateGraph(state_schema=Conditional_edge)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", Conditional_node)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

graph: CompiledStateGraph = builder.compile()

result = graph.invoke({"user_input": "What is FYP"})
# result = llm.invoke("What is FYP")
print(result)

# display(Image(graph.get_graph().draw_mermaid_png()))

