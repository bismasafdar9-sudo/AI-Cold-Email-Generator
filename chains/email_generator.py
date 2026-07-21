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

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise Exception(
        "GOOGLE_API_KEY not found. Please add it in Streamlit Secrets or .env"
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

    try:
        collection = client.get_collection(
            name="portfolio",
            embedding_function=embedding_function
        )

        if collection.count() > 0:
            return collection

    except Exception:
        pass

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

Write ONE professional cold email for the following job description.

Job Description:
{job_description}

Relevant Portfolio:
{portfolio}

Instructions:

- Use ONLY the portfolio information provided.
- Do NOT invent any projects, skills or experience.
- Mention the project naturally.
- Mention the relevant skills.
- Keep the email between 150 and 200 words.
- Use a professional tone.
- End with a polite closing.
""")

chain = prompt | llm

# ---------------- Generate Email ----------------


def generate_email(job_description):

    results = collection.query(
        query_texts=[job_description],
        n_results=1
    )

    if len(results["metadatas"][0]) == 0:
        return "No matching portfolio found."

    portfolio = results["metadatas"][0][0]

    response = chain.invoke({
        "job_description": job_description,
        "portfolio": portfolio
    })

    # If Gemini returns string
    if isinstance(response.content, str):
        return response.content

    # If Gemini returns list
    if isinstance(response.content, list):

        email = ""

        for item in response.content:

            if isinstance(item, dict):

                if item.get("type") == "text":
                    email += item.get("text", "")

            else:

                try:
                    email += item.text
                except Exception:
                    pass

        return email

    return str(response.content)