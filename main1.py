from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain, LLMChain

import os
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=os.getenv('huggingfacehub_api_token')
)

prompt1 = PromptTemplate(
    template="Translate the following text to Spanish: {text}", input_variables=["text"])
prompt2 = PromptTemplate(
    template="What is the sentiment of {text}", input_variables=["text"])

chain1 = LLMChain(llm=llm, prompt=prompt1)
chain2 = LLMChain(llm=llm, prompt=prompt2)
chain = SimpleSequentialChain(chains=[chain1, chain2])
response = chain.invoke("How are you?")

print(response)
