import os
import chromadb
from dotenv import load_dotenv

from chromadb.utils import embedding_functions
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# ---------------- Gemini ----------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

# ---------------- ChromaDB ----------------

client = chromadb.PersistentClient(path="chroma_db")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_collection(
    name="portfolio",
    embedding_function=embedding_function
)

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
- Do NOT invent any new projects, skills or experience.
- Mention the project name naturally.
- Mention the skills from the portfolio.
- Keep the email short (150–200 words).
- Write in a professional tone.
- End with a polite closing.
""")

chain = prompt | llm


def generate_email(job_description):

    # Search best matching project
    results = collection.query(
        query_texts=[job_description],
        n_results=1
    )

    portfolio = results["metadatas"][0][0]

    # Debug: Show retrieved project
    print("\nRetrieved Portfolio:")
    print(portfolio)

    # Generate email
    response = chain.invoke({
        "job_description": job_description,
        "portfolio": portfolio
    })

    # Handle Gemini response
    if isinstance(response.content, list):
        return "".join(
            part["text"]
            for part in response.content
            if isinstance(part, dict) and part.get("type") == "text"
        )

    return response.content