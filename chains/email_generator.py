import os
import streamlit as st
import chromadb
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ---------------- Load API Key ----------------

load_dotenv()

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise Exception(
        "GOOGLE_API_KEY not found. Add it in Streamlit Secrets or .env file."
    )

# ---------------- Gemini ----------------

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

# ---------------- ChromaDB ----------------

@st.cache_resource
def load_collection():

    client = chromadb.PersistentClient(path="chroma_db")

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    return client.get_collection(
        name="portfolio",
        embedding_function=embedding_function
    )

llm = load_llm()
collection = load_collection()

# ---------------- Prompt ----------------

prompt = ChatPromptTemplate.from_template("""
You are an expert cold email writer.

Write ONE professional cold email.

Job Description:
{job_description}

Relevant Portfolio:
{portfolio}

Instructions:
- Use ONLY the provided portfolio.
- Do NOT invent any skills or projects.
- Mention the project naturally.
- Mention relevant skills.
- Keep the email between 150-200 words.
- Professional tone.
- End politely.
""")

chain = prompt | llm

# ---------------- Generate Email ----------------

def generate_email(job_description):

    results = collection.query(
        query_texts=[job_description],
        n_results=1
    )

    if not results["metadatas"][0]:
        return "No matching portfolio found."

    portfolio = results["metadatas"][0][0]

    response = chain.invoke({
        "job_description": job_description,
        "portfolio": portfolio
    })

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        email = ""

        for item in content:

            if isinstance(item, dict):
                if item.get("type") == "text":
                    email += item.get("text", "")

            elif hasattr(item, "text"):
                email += item.text

        return email

    return str(content)