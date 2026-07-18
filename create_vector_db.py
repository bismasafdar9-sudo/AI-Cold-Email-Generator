import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# Read portfolio
df = pd.read_csv("portfolio.csv")

# Chroma Client
client = chromadb.PersistentClient(path="chroma_db")

# Local Embedding Model
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Delete old collection
try:
    client.delete_collection("portfolio")
except:
    pass

# Create Collection
collection = client.create_collection(
    name="portfolio",
    embedding_function=embedding_function
)

documents = []
metadatas = []
ids = []

for i, row in df.iterrows():
    text = f"""
    Project: {row['Project']}
    Skills: {row['Skills']}
    Description: {row['Description']}
    """

    documents.append(text)

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

print("✅ Vector Database Ready!")