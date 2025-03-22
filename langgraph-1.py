from typing_extensions import TypedDict
from langchain_core.messages.ai import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph  # type
from dotenv import load_dotenv
import os
load_dotenv()


class LearningState(TypedDict):
    prompt: str
    output: str

# zafeer_state: LearningState = LearningState(prompt= "hello from Zafeer Wains ")
# print(zafeer_state)


def node_1(state: LearningState) -> LearningState:
    print("---Node 1 State---", state)
    ai_msg: AIMessage = llm.invoke(state['prompt'])
    print("AI_MSG", ai_msg)
    return {"output": ai_msg.content}


def node_2(state: LearningState) -> LearningState:
    print("---Node 2 State---", state)
    ai_msg: AIMessage = llm.invoke(
        state['output'] + " Explain it In a way that it fells like a human is doing this an present that it is presented by Muhammad Zafeer Wains")
    return {"output": ai_msg.content}


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

builder: StateGraph = StateGraph(state_schema=LearningState)
builder.add_node("FIrst_Function", node_1)
builder.add_node("Second_Function", node_2)

builder.add_edge(START, "FIrst_Function")
builder.add_edge("FIrst_Function", "Second_Function")
builder.add_edge("Second_Function", END)

graph: CompiledStateGraph = builder.compile()

result=graph.invoke({"prompt": "What is LangChain"})

print(result)

