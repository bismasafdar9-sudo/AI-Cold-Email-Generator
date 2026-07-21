import os
import streamlit as st
import pandas as pd
import chromadb

from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ---------------- Load Environment ----------------

load_dotenv()

api_key = None

# First try Streamlit Secrets
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise Exception(
        "GOOGLE_API_KEY not found. Please add it in Streamlit Secrets."
    )

# ---------------- Gemini ----------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=api_key,
    temperature=0.3
)

# ---------------- ChromaDB ----------------

client = chromadb.PersistentClient(path="chroma_db")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def load_collection():
    """
    Load collection.
    If collection does not exist, create it automatically.
    """

    try:
        collection = client.get_collection(
            name="portfolio",
            embedding_function=embedding_function
        )

        if collection.count() > 0:
            return collection

    except Exception:
        pass

    # Delete old collection if exists
    try:
        client.delete_collection("portfolio")
    except Exception:
        pass

    collection = client.create_collection(
        name="portfolio",
        embedding_function=embedding_function
    )

    df = pd.read_csv("portfolio.csv")

    documents = []
    metadatas = []
    ids = []

    for i, row in df.iterrows():

        documents.append(
            f"""
Project: {row['Project']}
Skills: {row['Skills']}
Description: {row['Description']}
"""
        )

        metadatas.append({
            "project": row["Project"],
            "skills": row["Skills"],
            "link": row["Link"]
        })

        ids.append(str(i))

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    return collection


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
- Do NOT invent projects.
- Mention the retrieved project naturally.
- Mention relevant skills.
- Keep it between 150-200 words.
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

    if len(results["metadatas"][0]) == 0:
        return "No matching project found."

    portfolio = results["metadatas"][0][0]

    response = chain.invoke({
        "job_description": job_description,
        "portfolio": portfolio
    })

    if hasattr(response, "content"):
        return response.content

    return str(response)