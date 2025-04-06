from dotenv import load_dotenv
load_dotenv()

from load_vector_store import load_vector_store
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from hybrid_retriver import get_hybrid_retriever

# Use your upgraded hybrid retriever
retriever = get_hybrid_retriever()

# Use OpenAI LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.3)

# Improved prompt for query-aware summarization
prompt_template = ChatPromptTemplate.from_template("""
You are a smart assistant that reads news articles and returns a short summary relevant to a user query.

Task:
- ONLY summarize content highly relevant to this query: "{query}".
- Ignore or skip articles that are not relevant.
- Make the summary concise and useful (max 100 words).
- Do NOT repeat information across articles.
- Write in a neutral, informative tone.

Articles:
{documents}

Summary:
""")

def format_docs(docs):
    return "\n\n".join(
        f"Headline: {doc.metadata.get('headline', '[No Headline]')}\n"
        f"Summary: {doc.metadata.get('short_description') or doc.page_content[:200]}\n"
        f"Link: {doc.metadata.get('link', '[No Link]')}"
        for doc in docs
    )


# LLM chain with upgraded prompt
chain = (
    RunnableLambda(lambda query: {"query": query, "documents": format_docs(retriever(query))})
    | prompt_template
    | llm
)

def answer_query(query: str):
    docs = retriever(query)[:3]  # Already re-ranked + boosted
    summary = chain.invoke(query)
    return summary.content.strip(), docs
