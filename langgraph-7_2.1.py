from typing_extensions import TypedDict
from typing import Literal
import random
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))


class TypedDictState(TypedDict):
    foo: str
    bar: str


choco_bars: TypedDictState = TypedDictState(company="Choco", bar="M&Ms")
print(choco_bars["bar"])
print(choco_bars["company"])


class TypedDictState(TypedDict):
    name: str
    mood: Literal["happy", "sad"]


override_mood: TypedDictState = TypedDictState(
    name="Lance", mood="mad", random_field="user")
override_mood["mood"]
print(override_mood)


def node_1(state: TypedDictState):
    print("---Node 1---")
    return {"name": state['name'] + " is ... "}


def node_2(state: TypedDictState):
    print("---Node 2---")
    return {"mood": "happy"}


def node_3(state: TypedDictState):
    print("---Node 3---")
    return {"mood": "sad"}


def decide_mood(state: TypedDictState) -> Literal["node_2", "node_3"]:

    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"

    # 50% of the time, we return Node 3
    return "node_3"


# Build graph
builder: StateGraph = StateGraph(TypedDictState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph: CompiledStateGraph = builder.compile()

print(graph.invoke({"name":"Lance"}))