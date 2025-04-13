import os
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = "all-MiniLM-L6-v2"
QDRANT_URL = os.environ.get("QDRANT_URL")
OPENAI_KEY = os.environ.get("OPENAI_KEY")
CONTEXT_SYSTEM_PROMPT = """
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

SUMMARY_SYSTEM_PROMPT = """You are an expert summarizer

Summarize the entire conversation between the user and the assistant, including greetings and any follow-up questions.

Write it in second-person as if you are the assitant speaking directly to the user.

Respond ONLY with a JSON object in this exact format:
{
  "answer": "your summary goes here"
}"""
# SUMMARY_SYSTEM_PROMPT = """
#         You are an expert summarizer. Please read through the conversation below between a user and an assistant and generate a concise summary that captures the key points, topics, and main messages discussed, try your best to focus on all messages rather than the last.
#         Focus on creating a clear overview that includes the user's questions and the assistant's responses. Output only the summary text.
#         The conversation provided below is structured as a list of messages in JSON format.
#         Each message has a "role" field (either "user" or "assistant") and a "content" field with the text of the message.
#         your summary should be in a first-person tone, as if your are speaking to the user.
#          Provide an answer in the following JSON format:
#         {
#             "answer": string
#         }
#         """
