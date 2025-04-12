import json

from openai import OpenAI


class Chatbot:

    @classmethod
    def search(cls, openai_client: OpenAI, data, message):
        system_prompt = cls.build_system_prompt(data)
        system_message = {"role": "system", "content": system_prompt}
        user_message = {"role": "user", "content": message}

        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[system_message, user_message],
            response_format={"type": "json_object"},
        )

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

        system_prompt = """You are an expert summarizer. 
        Summarize the following conversation between a user and a Start-Ups related Q&A AI Chatbot. 
        Your summary should capture the key topics, questions, and responses in a concise and coherent manner.
        your summary should be in a first-person tone, as if your are speaking to the user
        Provide an answer in the following JSON format:
        {
            "answer": string
        }
        """
        formatted_data = "\n-----\n".join(
            [
                f"Qeustion: {message['message']}\nAnswer: {message['answer']}\n"
                for message in user_history
            ]
        )
        system_message = {"role": "system", "content": system_prompt}
        user_message = {"role": "user", "content": formatted_data}

        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[system_message, user_message],
            response_format={"type": "json_object"},
        )

        answer = json.loads(completion.choices[0].message.content)["answer"]
        return answer
        