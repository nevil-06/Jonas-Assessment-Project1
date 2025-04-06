# build_vector_store.py

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import os

# File paths
DATA_FILE = "cleaned_news_final.csv"
INDEX_PATH = "faiss_index"

# Create index directory if not present
os.makedirs(INDEX_PATH, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE)

# Convert rows to LangChain Documents
retrieval_prefix = "Represent this sentence for retrieval: "
docs = [
    Document(
        page_content=retrieval_prefix + row['text'],
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

# 👉 Use Sentence-BERT: all-MiniLM-L6-v2
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Build vector store
print("🔄 Creating FAISS index with Sentence-BERT embeddings...")
vector_store = FAISS.from_documents(docs, embedding_model)
vector_store.save_local(INDEX_PATH)
print(f"✅ FAISS index saved at {INDEX_PATH}")
