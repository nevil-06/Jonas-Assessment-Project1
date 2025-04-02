# hybrid_retriever.py

from langchain_community.retrievers import BM25Retriever
from load_vector_store import load_vector_store
from langchain.retrievers import EnsembleRetriever
# Prepare keyword retriever (BM25)
from langchain_community.document_loaders import CSVLoader
from langchain.docstore.document import Document

import pandas as pd
# Load FAISS (semantic retriever)
vector_store = load_vector_store()
semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 3})



# Load CSV and convert to Document format
df = pd.read_csv("cleaned_news_final.csv")
# documents = [Document(page_content=row["text"]) for _, row in df.iterrows()]
documents = [
    Document(
        page_content=row["text"],
        metadata={
            "id": row["id"],
            "category": row["category"],
            "headline": row["headline"],
            "short_description": row["short_description"],
            "link": row["link"]
        }
    )
    for _, row in df.iterrows()
]

# Build BM25 retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3

# Combine both using ensemble retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.6, 0.4],  # Tune this if needed
)

def get_hybrid_retriever():
    return hybrid_retriever