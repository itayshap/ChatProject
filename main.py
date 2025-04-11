import uvicorn
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

from config import OPENAI_KEY
from services.chatbot_service import Chatbot
from services.neural_search_service import NeuralSearcher

app = FastAPI()
neural_searcher = NeuralSearcher(collection_name="startups")
openai_client = OpenAI(api_key=OPENAI_KEY)


class Query(BaseModel):
    message: str


@app.post("/query")
async def query(query: Query):
    retrieved_data = neural_searcher.search(text=query.message)
    output = Chatbot.search(openai_client, retrieved_data, query.message)
    return {"output": output}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
