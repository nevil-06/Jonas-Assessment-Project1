# load_vector_store.py

from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings



def load_vector_store(index_path="faiss_index"):
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = FAISS.load_local(
    index_path,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
    return vector_store