from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.getenv("Groq_API_KEY")
model=ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

### Prompt Templates
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

parser=StrOutputParser()

#Create Chain
chain=prompt_template|model|parser

## App Defination
app=FastAPI(title="Langchain Server",
            version="1.0",
            description="A simple API server using Langchain runnable interfaces",)


## Adding chain routes
add_routes(
    app,
    chain,
    path="/chain"
)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1", port=8000)