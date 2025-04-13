from fastapi.responses import JSONResponse
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from openai import OpenAI
from pydantic import BaseModel
from collections import deque

from config import OPENAI_KEY
from services.chatbot_service import Chatbot
from services.neural_search_service import NeuralSearcher

OUTDATED_QUERY = "Your query has been updated. Please wait for the latest answer."

def get_user_id_from_header(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> str:
    """
    Retrieves the user_id from the Authorization header.
    Raises an HTTPException if the header is missing or invalid.
    """
    user_id = credentials.credentials
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID missing in Authorization header")
    return user_id

def has_pending_messages(user_id: str):
    if user_pending_messages[user_id] > 0:
        user_pending_messages[user_id] -= 1
        return True
    return False

app = FastAPI(dependencies=[Depends(get_user_id_from_header)])

neural_searcher = NeuralSearcher(collection_name="startups")
openai_client = OpenAI(api_key=OPENAI_KEY)

user_history = {}
user_pending_messages = {}

class Query(BaseModel):
    message: str

@app.post("/query")
async def query(query: Query, user_id: str = Depends(get_user_id_from_header)):
    user_message = query.message
    
    if user_id not in user_history:
        user_history[user_id] = deque(maxlen=10)
    if user_id not in user_pending_messages:
        user_pending_messages[user_id] = 0
    else:
        user_pending_messages[user_id] += 1

    user_history[user_id].append({"role": "user", "content": user_message})

    if len(user_history[user_id]) > 0:
        user_message = await Chatbot.build_user_message(openai_client, user_message, list(user_history[user_id]))
        if has_pending_messages(user_id):
            return {"output": OUTDATED_QUERY}

    retrieved_data = await neural_searcher.search(text=user_message)
    if has_pending_messages(user_id):
        return {"output": OUTDATED_QUERY}

    output = await Chatbot.search(openai_client, retrieved_data, user_message)
    if has_pending_messages(user_id):
        return {"output": OUTDATED_QUERY}

    user_pending_messages.pop(user_id)
    user_history[user_id].append({"role": "assistant", "content": output})

    return JSONResponse(content={"output": output})

@app.get("/summarize")
async def summarize(user_id: str = Depends(get_user_id_from_header)):
    if user_id not in user_history:
        return {"output": "No History"}
    output = Chatbot.summarize(list(user_history[user_id]), openai_client)
    return {"output": output}

@app.get("/history")
async def history(user_id: str = Depends(get_user_id_from_header)):
    if user_id not in user_history:
        return {"output": "No History"}
    return {"output": user_history[user_id]}
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
