import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_ollama import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.chat import (
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)

class ChatRequest(BaseModel):
    input: str

class ChatResponse(BaseModel):
    reply: str

app = FastAPI(title="Chat Microservice")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Build the chain once
model_name = os.getenv("CHAT_MODEL", "gemma3:1b")
llm = ChatOllama(model=model_name)
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

system_prompt = os.getenv("SYSTEM_PROMPT",
    """You are an expert assistant who knows multiple languages that always checks whether each command makes sense given the previous context, and only responds if it does. 
    Make responses short and informative. Use only simple punctuation like . , ! ? and no emojis.
     If the user writes in English, respond in English.
If the user writes in Spanish, respond in Spanish.
If the user writes in French, respond in French.
And so on for any other language. Do not mix languages in your response.""")
system_tmpl = SystemMessagePromptTemplate.from_template(system_prompt)

chat_prompt = ChatPromptTemplate.from_messages([
    system_tmpl,
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])
chat_chain = ConversationChain(llm=llm, memory=memory, prompt=chat_prompt)

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    reply = chat_chain.predict(input=req.input)
    return {"reply": reply}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chat_service:app", host="0.0.0.0", port=int(os.getenv("PORT",8003)), reload=True)
