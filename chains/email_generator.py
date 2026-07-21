import os
import pandas as pd
import chromadb

from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------- Gemini ----------------

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise Exception(
        "GOOGLE_API_KEY not found. Add it in Streamlit Secrets or .env file."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.3,
)

# ---------------- ChromaDB ----------------

client = chromadb.PersistentClient(path="chroma_db")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def get_or_create_collection():
    try:
        collection = client.get_collection(
            name="portfolio",
            embedding_function=embedding_function,
        )

        count = collection.count()

        if count > 0:
            return collection

    except:
        pass

    # Create new collection
    try:
        client.delete_collection("portfolio")
    except:
        pass

    collection = client.create_collection(
        name="portfolio",
        embedding_function=embedding_function,
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

        metadatas.append(
            {
                "project": row["Project"],
                "skills": row["Skills"],
                "link": row["Link"],
            }
        )

        ids.append(str(i))

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    return collection


collection = get_or_create_collection()

# ---------------- Prompt ----------------

prompt = ChatPromptTemplate.from_template(
    """
You are an expert cold email writer.

Write ONE professional cold email.

Job Description:
{job_description}

Relevant Portfolio:
{portfolio}

Instructions:

- Mention ONLY the retrieved project.
- Mention ONLY the retrieved skills.
- Do NOT invent any experience.
- Keep it between 150 and 200 words.
- Professional tone.
- End politely.
"""
)

chain = prompt | llm


def generate_email(job_description):

    results = collection.query(
        query_texts=[job_description],
        n_results=1,
    )

    if len(results["metadatas"][0]) == 0:
        return "No matching portfolio found."

    portfolio = results["metadatas"][0][0]

    response = chain.invoke(
        {
            "job_description": job_description,
            "portfolio": portfolio,
        }
    )

    return response.content