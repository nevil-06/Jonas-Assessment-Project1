# load_vector_store.py

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def load_vector_store(index_path="faiss_index"):
    vector_store = FAISS.load_local(
        index_path, embeddings=embedding_model, allow_dangerous_deserialization=True
    )
    return vector_store
