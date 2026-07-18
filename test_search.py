import chromadb
from chromadb.utils import embedding_functions

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

# Same embedding model
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Load collection
collection = client.get_collection(
    name="portfolio",
    embedding_function=embedding_function
)

# Search
results = collection.query(
    query_texts=["Need Flask Developer"],
    n_results=1
)

print("\n✅ Best Match:\n")
print(results["metadatas"][0][0])