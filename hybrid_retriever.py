# import pandas as pd
# from langchain_community.retrievers import BM25Retriever
# from langchain.docstore.document import Document
# from load_vector_store import load_vector_store
# from ner_utils import extract_named_entities
# from sentence_transformers import CrossEncoder

# # Load the CrossEncoder model (lightweight and effective)
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# def boost_named_entities(query, docs):
#     """
#     Boost docs containing relevant named entities from the query.
#     """
#     relevant_labels = ("GPE", "ORG", "PERSON", "EVENT", "DATE", "NORP", "LOC")
#     entities = [ent.lower() for ent, label in extract_named_entities(query) if label in relevant_labels]

#     boosted = []
#     for doc in docs:
#         content = doc.page_content.lower()
#         boost_score = sum(ent in content for ent in entities)
#         doc.metadata["boost_score"] = boost_score
#         boosted.append(doc)

#     return boosted

# def rerank_with_crossencoder(query, docs, top_n=3):
#     pairs = [[query, doc.page_content] for doc in docs]
#     scores = cross_encoder.predict(pairs)

#     scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
#     return [doc for doc, _ in scored_docs[:top_n]]

# def custom_hybrid_retrieve(query, k=3, bm25_weight=0.4, semantic_weight=0.6):
#     # Load FAISS vector store
#     vector_store = load_vector_store()
#     semantic_docs = vector_store.similarity_search(query, k=5)

#     # Load full dataset for BM25
#     df = pd.read_csv("cleaned_news_final.csv")
#     documents = [
#         Document(
#             page_content=row["text"],
#             metadata={
#                 "id": row["id"],
#                 "category": row["category"],
#                 "headline": row["headline"],
#                 "short_description": row["short_description"],
#                 "link": row["link"]
#             }
#         )
#         for _, row in df.iterrows()
#     ]
#     bm25 = BM25Retriever.from_documents(documents)
#     bm25.k = 5
#     bm25_docs = bm25.invoke(query)

#     # Score and merge
#     scores = {}
#     for rank, doc in enumerate(bm25_docs):
#         doc_id = doc.metadata.get("id")
#         scores[doc_id] = scores.get(doc_id, 0) + bm25_weight * (1 / (rank + 1))

#     for rank, doc in enumerate(semantic_docs):
#         doc_id = doc.metadata.get("id")
#         scores[doc_id] = scores.get(doc_id, 0) + semantic_weight * (1 / (rank + 1))

#     # Combine all docs
#     all_docs = {doc.metadata["id"]: doc for doc in bm25_docs + semantic_docs}
#     ranked = sorted(all_docs.items(), key=lambda x: scores.get(x[0], 0), reverse=True)
#     top_docs = [doc for _, doc in ranked[:15]]

#     # Apply NER boosting
#     boosted = boost_named_entities(query, top_docs)

#     # Final Re-ranking using CrossEncoder
#     return rerank_with_crossencoder(query, boosted, top_n=k)

# def get_hybrid_retriever():
#     return lambda query: custom_hybrid_retrieve(query)



import pandas as pd
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document
from load_vector_store import load_vector_store
from ner_utils import extract_named_entities
from sentence_transformers import CrossEncoder

# ✅ Load resources once at the module level
vector_store = load_vector_store()
cross_encoder = CrossEncoder("cross-encoder/stsb-TinyBERT-L-4")  # Fast & accurate

# ✅ Load BM25 documents once
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
bm25.k = 5  # Smaller top-k for speed

def boost_named_entities(query, docs):
    """
    Boost documents that contain key entities from the query.
    """
    relevant_labels = ("GPE", "ORG", "PERSON", "EVENT", "DATE", "NORP", "LOC")
    entities = [ent.lower() for ent, label in extract_named_entities(query) if label in relevant_labels]

    boosted = []
    for doc in docs:
        content = doc.page_content.lower()
        boost_score = sum(ent in content for ent in entities)
        doc.metadata["boost_score"] = boost_score
        boosted.append(doc)

    return boosted

def rerank_with_crossencoder(query, docs, top_n=3):
    pairs = [[query, doc.page_content] for doc in docs]
    scores = cross_encoder.predict(pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_n]]

def custom_hybrid_retrieve(query, k=3, bm25_weight=0.4, semantic_weight=0.6):
    # Retrieve from both BM25 and FAISS
    semantic_docs = vector_store.similarity_search(query, k=5)
    bm25_docs = bm25.invoke(query)

    # Weighted score merging
    scores = {}
    for rank, doc in enumerate(bm25_docs):
        doc_id = doc.metadata.get("id")
        scores[doc_id] = scores.get(doc_id, 0) + bm25_weight * (1 / (rank + 1))

    for rank, doc in enumerate(semantic_docs):
        doc_id = doc.metadata.get("id")
        scores[doc_id] = scores.get(doc_id, 0) + semantic_weight * (1 / (rank + 1))

    # Combine and rank all unique docs
    all_docs = {doc.metadata["id"]: doc for doc in bm25_docs + semantic_docs}
    ranked = sorted(all_docs.items(), key=lambda x: scores.get(x[0], 0), reverse=True)
    top_docs = [doc for _, doc in ranked[:15]]

    # Apply NER-based boosting
    boosted = boost_named_entities(query, top_docs)

    # Final rerank with CrossEncoder
    return rerank_with_crossencoder(query, boosted, top_n=k)

def get_hybrid_retriever():
    return lambda query: custom_hybrid_retrieve(query)
