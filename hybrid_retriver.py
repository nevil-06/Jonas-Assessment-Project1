# # hybrid_retriever.py

# from langchain_community.retrievers import BM25Retriever
# from load_vector_store import load_vector_store
# from langchain.retrievers import EnsembleRetriever
# # Prepare keyword retriever (BM25)
# from langchain_community.document_loaders import CSVLoader
# from langchain.docstore.document import Document

# import pandas as pd
# # Load FAISS (semantic retriever)
# vector_store = load_vector_store()
# semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 3})



# # Load CSV and convert to Document format
# df = pd.read_csv("cleaned_news_final.csv")
# # documents = [Document(page_content=row["text"]) for _, row in df.iterrows()]
# documents = [
#     Document(
#         page_content=row["text"],
#         metadata={
#             "id": row["id"],
#             "category": row["category"],
#             "headline": row["headline"],
#             "short_description": row["short_description"],
#             "link": row["link"]
#         }
#     )
#     for _, row in df.iterrows()
# ]

# # Build BM25 retriever
# bm25_retriever = BM25Retriever.from_documents(documents)
# bm25_retriever.k = 3

# # Combine both using ensemble retriever
# hybrid_retriever = EnsembleRetriever(
#     retrievers=[bm25_retriever, semantic_retriever],
#     weights=[0.6, 0.4],  # Tune this if needed
# )

# def get_hybrid_retriever():
#     return hybrid_retriever



# hybrid_retriever.py

from langchain_community.retrievers import BM25Retriever
from load_vector_store import load_vector_store
from langchain.docstore.document import Document
import pandas as pd

def custom_hybrid_retrieve(query, k=3, bm25_weight=0.4, semantic_weight=0.6):
    # Load vector store and retriever
    vector_store = load_vector_store()
    semantic_docs = vector_store.similarity_search(query, k=5)

    # Load dataset and build BM25 retriever
    df = pd.read_csv("cleaned_news_final.csv")
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
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = 10
    # bm25_docs = bm25.get_relevant_documents(query)
    bm25_docs = bm25.invoke(query)


    # Score and combine documents
    scores = {}
    for rank, doc in enumerate(bm25_docs):
        doc_id = doc.metadata.get("id")
        scores[doc_id] = scores.get(doc_id, 0) + bm25_weight * (1 / (rank + 1))

    for rank, doc in enumerate(semantic_docs):
        doc_id = doc.metadata.get("id")
        scores[doc_id] = scores.get(doc_id, 0) + semantic_weight * (1 / (rank + 1))

    # Merge and rank
    all_docs = {doc.metadata["id"]: doc for doc in bm25_docs + semantic_docs}
    ranked = sorted(all_docs.items(), key=lambda x: scores.get(x[0], 0), reverse=True)
    
    return [doc for _, doc in ranked[:k]]

def get_hybrid_retriever():
    return lambda query: custom_hybrid_retrieve(query)