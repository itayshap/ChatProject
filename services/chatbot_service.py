import json
from asyncio import sleep

from openai import OpenAI


class Chatbot:

    @classmethod
    async def search(cls, openai_client: OpenAI, data, message):
        system_prompt = cls.build_system_prompt(data)
        system_message = {"role": "system", "content": system_prompt}
        user_message = {"role": "user", "content": message}

        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[system_message, user_message],
            response_format={"type": "json_object"},
        )
        await sleep(5)
        answer = json.loads(completion.choices[0].message.content)["answer"]
        return answer

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
    def summarize(user_history: list, openai_client: OpenAI):
        messages = []
        system_prompt = """
        You are an expert summarizer. Please read through the conversation below between a user and an assistant and generate a concise summary that captures the key points, topics, and main messages discussed. 
        Focus on creating a clear overview that includes the user's questions and the assistant's responses. Output only the summary text.
        The conversation provided below is structured as a list of messages in JSON format.
        Each message has a "role" field (either "user" or "assistant") and a "content" field with the text of the message.
        your summary should be in a first-person tone, as if your are speaking to the user.
         Provide an answer in the following JSON format:
        {
            "summary": string
        }       
        """
        system_message = {"role": "system", "content": system_prompt}
        messages.append(system_message)
        messages += user_history
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
        )
        answer = json.loads(completion.choices[0].message.content)["summary"]
        return answer
    
    @classmethod
    async def build_user_message(cls, openai_client: OpenAI, message, user_history):
        messages = []
        system_prompt = """
        You are a question clarifier. I will provide you with a conversation history structured as individual messages.
        Each user message is marked with "role": "user" and each assistant response is marked with "role": "assistant".
        Your task is to rephrase an ambiguous follow-up question into a fully self-contained query by integrating the relevant context from the conversation history.
        For example, if the conversation history contains:
        User: "Are there startups about wine in Chicago?"
        Assistant: "Yes, Winestyr is a startup based in Chicago that offers a smarter way to wine by providing an alternative to mass-produced wines."
        and the ambiguous follow-up question is:
        "And in NY?"
        then you should output:
        "Are there startups about wine in NY?"
        If there is no conversation history or you determine that the follow-up question does not depend on prior context, return the follow-up question unchanged.
        Please output only the clarified, fully formed question in the following JSON format:
        {
            "answer": string
        }
        """
        
        system_message = {"role": "system", "content": system_prompt}
        messages.append(system_message)
        messages += user_history
        messages.append({"role": "user", "content": message})
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
        )
        await sleep(5)
        answer = json.loads(completion.choices[0].message.content)["answer"]
        return answer
    
    @staticmethod
    def build_user_prompt(user_history):
        formatted_data  = []
        for message in user_history:
            formatted_data.append({"role": "user", "content": message['message']})
            formatted_data.append({"role": "assistant", "content": message['answer']})
        
        return formatted_data

