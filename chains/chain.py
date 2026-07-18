import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


class Chain:

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.5
        )

    def write_mail(self, job_description, portfolio):

        prompt = PromptTemplate.from_template("""
You are an expert cold email writer.

Write a professional cold email for the following job description.

Job Description:
{job_description}

Relevant Portfolio:
{portfolio}

Rules:
- Keep the email short and professional.
- Mention the relevant projects naturally.
- End politely.
""")

        chain = prompt | self.llm

        return chain.invoke({
            "job_description": job_description,
            "portfolio": portfolio
        })