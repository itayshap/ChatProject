from fastapi.responses import JSONResponse
import uvicorn
from fastapi import FastAPI, Request, Response
from openai import OpenAI
from pydantic import BaseModel
import uuid

from config import OPENAI_KEY
from services.chatbot_service import Chatbot
from services.neural_search_service import NeuralSearcher

app = FastAPI()
neural_searcher = NeuralSearcher(collection_name="startups")
openai_client = OpenAI(api_key=OPENAI_KEY)

user_history = {}

class Query(BaseModel):
    message: str

@app.middleware("http")
async def add_user_id_cookie(request: Request, call_next):
    response = await call_next(request)
    user_id = request.cookies.get("user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
        response.set_cookie(key="user_id", value=user_id, httponly=True)
    return response

@app.post("/query")
async def query(query: Query, request: Request, response: Response):
    user_id = request.cookies.get("user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
    if user_id not in user_history:
        user_history[user_id] = []
    response.set_cookie(key="user_id", value=user_id, httponly=True)
    retrieved_data = neural_searcher.search(text=query.message)
    output = Chatbot.search(openai_client, retrieved_data, query.message)
    user_history[user_id].append({"message" : query.message, "answer" : output})
    if len(user_history[user_id]) > 10:
        user_history[user_id] = user_history[user_id][-10:]
    response = JSONResponse(content={"output": output})
    return response

@app.get("/summarize")
async def summarize(request: Request):
    user_id = request.cookies.get("user_id")
    if not user_id or user_id not in user_history:
        return {"output": "No History"}
    output = Chatbot.summarize(user_history[user_id], openai_client)
    return ({"output" : output})
     
@app.get("/history")
async def history(request: Request):
    user_id = request.cookies.get("user_id")
    if not user_id or user_id not in user_history:
        return {"output": "No History"}
    return {"output": user_history[user_id]}
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)
