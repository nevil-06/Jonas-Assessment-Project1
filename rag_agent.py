# rag_agent.py

from dotenv import load_dotenv
load_dotenv()

from load_vector_store import load_vector_store
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from hybrid_retriver import get_hybrid_retriever

# Load FAISS vector store
vector_store = load_vector_store()
retriever = get_hybrid_retriever()

# Load OpenAI LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

# Prompt for summarizing documents
prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant. Summarize the following news articles related to "{query}":

{documents}

Your summary should be clear, concise, and useful for a general reader.
""")

# Format retrieved docs into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define custom summarization chain
chain = (
    RunnableLambda(lambda query: {"query": query, "documents": format_docs(retriever.invoke(query))})
    | prompt_template
    | llm
)

def answer_query(query: str):
    """Return a summary and the retrieved documents."""
    docs = retriever.invoke(query)
    docs = docs[:3]
    summary = chain.invoke(query)
    return summary.content, docs