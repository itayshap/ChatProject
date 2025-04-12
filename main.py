import asyncio
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
user_sequential_messages = {}

class Query(BaseModel):
    message: str

def get_or_create_user_id(request: Request, response: Response) -> str:
    """
    Retrieves the user_id from cookies or creates a new one if it doesn't exist.
    Sets the user_id cookie in the response if a new one is created.
    """
    user_id = request.cookies.get("user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
        response.set_cookie(key="user_id", value=user_id, httponly=True)
    return user_id

@app.post("/query")
async def query(query: Query, request: Request, response: Response):
    user_message = query.message
    user_id = get_or_create_user_id(request, response)
    
    if user_id not in user_history:
        user_history[user_id] = []
    if user_id not in user_sequential_messages:
        user_sequential_messages[user_id] = 0
    else:
        user_sequential_messages[user_id] += 1

    print(f"this is the sequential: {user_sequential_messages[user_id]}")

    if len(user_history[user_id]) > 0:
        task = asyncio.create_task(Chatbot.search_with_context(openai_client, user_message, user_history[user_id]))
        user_message = await task

    retrieved_data = neural_searcher.search(text=user_message)
    task1 = asyncio.create_task(Chatbot.search(openai_client, retrieved_data, user_message))
    output = await task1

    user_history[user_id].append({"role": "user", "content": user_message})
    if user_sequential_messages[user_id] > 0:
        user_sequential_messages[user_id] -= 1
        outdated_response = "Your query has been updated. Please wait for the latest answer."
        return {"output": outdated_response}

    user_sequential_messages.pop(user_id)
    user_history[user_id].append({"role": "assistant", "content": output})

    if len(user_history[user_id]) > 20:
        user_history[user_id] = user_history[user_id][-20:]

    response = JSONResponse(content={"output": output})
    return response

@app.get("/summarize")
async def summarize(request: Request, response: Response):
    user_id = get_or_create_user_id(request, response)
    if user_id not in user_history:
        return {"output": "No History"}
    output = Chatbot.summarize(user_history[user_id], openai_client)
    return {"output": output}

@app.get("/history")
async def history(request: Request, response: Response):
    user_id = get_or_create_user_id(request, response)
    if user_id not in user_history:
        return {"output": "No History"}
    return {"output": user_history[user_id]}
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)
