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

@app.post("/query")
async def query(query: Query, request: Request, response: Response):
    user_message = query.message
    user_id = request.cookies.get("user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
    if user_id not in user_history:
        user_history[user_id] = []
    response.set_cookie(key="user_id", value=user_id, httponly=True)
    if len(user_history[user_id]) > 0:
        user_message = Chatbot.search2(openai_client, user_message, user_history[user_id])
        print(user_message)
    retrieved_data = neural_searcher.search(text=user_message)
    output = Chatbot.search(openai_client, retrieved_data, user_message)
    user_history[user_id].append({"role" : "user", "content": user_message})
    user_history[user_id].append({"role" : "assistant", "content": output})
    if len(user_history[user_id]) > 20:
        user_history[user_id] = user_history[user_id][-20:]
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
