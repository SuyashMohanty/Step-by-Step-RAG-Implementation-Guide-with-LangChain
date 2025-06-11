import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv() #load all the enviroment variables

#Langsmith Tracking
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") 
os.environ["LANGSMITH_TRACING"]= "true"
os.environ["LANGSMITH_ENDPOINT"]= "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

## Prompt Tempelate
prompt=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked"),
        ("user", "Question:{question}")
    ]
)

## streamlit framework
st.title("Langchain Demo With Ollama Gemma2b")
input_text=st.text_input("What question you have in mind?")

##Ollama gemma:2b model
llm=OllamaLLM(model="gemma:2b")
output_parser= StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))