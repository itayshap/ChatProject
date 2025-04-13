import json
from asyncio import sleep
from config import SUMMARY_SYSTEM_PROMPT, CONTEXT_SYSTEM_PROMPT
from openai import OpenAI


class Chatbot:

    @classmethod
    async def search(cls, openai_client: OpenAI, data, message):
        system_prompt = cls.build_system_prompt(data)
        messages = cls.prepare_model_messages(system_prompt, [{"role": "user", "content": message}])
        return cls.run_model(openai_client, messages)

    @classmethod
    def summarize(cls, user_history: list, openai_client: OpenAI):
        messages = cls.prepare_model_messages(SUMMARY_SYSTEM_PROMPT, user_history)
        return cls.run_model(openai_client, messages)

    @classmethod
    async def build_user_message(cls, openai_client: OpenAI, message, user_history):
        messages = cls.prepare_model_messages(CONTEXT_SYSTEM_PROMPT, user_history)
        messages.append({"role": "user", "content": message})
        return cls.run_model(openai_client, messages)

    @staticmethod
    def build_system_prompt(retrieved_data):
        formatted_data = "\n-----\n".join(
            [
                f"Name: {startup['name']}\nCity: {startup['city']}\nDescription:{startup['description']}"
                for startup in retrieved_data
            ]
        )
        return f"""You are a Q&A chatbot that answers questions about a start-ups.
Keep your answers short and to the point.
Your whole knowledge is the following information:

###
{formatted_data}
###

Provide an answer in the following JSON format:
{{
    "answer": string
}}
"""

    @staticmethod
    def build_user_prompt(user_history: list):
        formatted_data  = []
        for message in user_history:
            formatted_data.append({"role": "user", "content": message['message']})
            formatted_data.append({"role": "assistant", "content": message['answer']})
        
        return formatted_data

    @staticmethod
    def run_model(openai_client : OpenAI, messages : list):
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
        )
        answer = json.loads(completion.choices[0].message.content)["answer"]
        return answer

    @staticmethod
    def prepare_model_messages(system_prompt : str, user_prompt : list[dict]):
        messages = []
        system_message = {"role": "system", "content": system_prompt}
        messages.append(system_message)
        messages += user_prompt
        return messages
    