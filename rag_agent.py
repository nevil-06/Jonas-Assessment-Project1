# from dotenv import load_dotenv
# load_dotenv()

# from load_vector_store import load_vector_store
# from langchain_openai import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableLambda
# from hybrid_retriver import get_hybrid_retriever
# import requests
# from bs4 import BeautifulSoup

# def extract_article_text(link):
#     """
#     Attempts to scrape the full article text from a URL.
#     Returns None if failed or no usable content found.
#     """
#     try:
#         response = requests.get(link, timeout=5)
#         if response.status_code == 200:
#             soup = BeautifulSoup(response.text, "html.parser")
#             paragraphs = soup.find_all("p")
#             article_text = " ".join(p.get_text() for p in paragraphs)
#             return article_text.strip()[:1000]  # Limit to 1000 chars
#     except Exception as e:
#         print(f"[Warning] Failed to fetch article from {link}\n{e}")
#     return None

# def format_docs(docs):
#     """
#     Format documents for summarization.
#     Priority: Full article (scraped) > short_description > page_content[:300]
#     """
#     formatted = []
#     for doc in docs:
#         link = doc.metadata.get("link", "")
#         short_desc = doc.metadata.get("short_description")
#         page_text = doc.page_content[:300]
#         headline = doc.metadata.get("headline", "[No Headline]")

#         # Try to scrape the full article
#         full_text = extract_article_text(link)

#         if full_text:
#             content = full_text
#         elif short_desc and isinstance(short_desc, str) and len(short_desc.strip()) > 0:
#             content = short_desc.strip()
#         else:
#             content = page_text.strip()

#         formatted.append(
#             f"Headline: {headline}\n"
#             f"Content: {content}\n"
#             f"Link: {link}"
#         )

#     return "\n\n".join(formatted)


# # Use your upgraded hybrid retriever
# retriever = get_hybrid_retriever()

# # Use OpenAI LLM
# llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.3)

# # Improved prompt for query-aware summarization
# prompt_template = ChatPromptTemplate.from_template("""
# You are a smart assistant that reads news articles and returns a short summary relevant to a user query.

# Task:
# - ONLY summarize content highly relevant to this query: "{query}".
# - Ignore or skip articles that are not relevant.
# - Make the summary concise and useful (max 100 words).
# - Do NOT repeat information across articles.
# - Write in a neutral, informative tone.

# Articles:
# {documents}

# Summary:
# """)

# def format_docs(docs):
#     return "\n\n".join(
#         f"Headline: {doc.metadata.get('headline', '[No Headline]')}\n"
#         f"Summary: {doc.metadata.get('short_description') or doc.page_content[:200]}\n"
#         f"Link: {doc.metadata.get('link', '[No Link]')}"
#         for doc in docs
#     )


# # LLM chain with upgraded prompt
# chain = (
#     RunnableLambda(lambda query: {"query": query, "documents": format_docs(retriever(query))})
#     | prompt_template
#     | llm
# )

# def answer_query(query: str):
#     docs = retriever(query)[:3]  # Already re-ranked + boosted
#     summary = chain.invoke(query)
#     return summary.content.strip(), docs


from dotenv import load_dotenv
load_dotenv()

import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from hybrid_retriever import get_hybrid_retriever

# Load retriever and LLM
retriever = get_hybrid_retriever()
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.3)

# Improved summarization prompt
prompt_template = ChatPromptTemplate.from_template("""
You are a smart assistant that reads news articles and returns a short summary relevant to a user query.

Task:
- ONLY summarize content highly relevant to this query: "{query}".
- Ignore or skip articles that are not relevant.
- Make the summary concise and useful (max 150 words).
- Do NOT repeat information across articles.
- Write in a neutral, informative tone.

Articles:
{documents}

Summary:
""")

# Extract full article content from a link if available
def extract_article_text(link):
    try:
        response = requests.get(link, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            article_text = " ".join(p.get_text() for p in paragraphs)
            print(article_text, "<<------>>")
            return article_text.strip()[:1000]  # Cap to 1000 characters
    except Exception as e:
        print(f"[Warning] Failed to fetch article from {link}\n{e}")
    return None

# Format documents for LLM summarization
def format_docs(docs):
    formatted = []
    for doc in docs:
        link = doc.metadata.get("link", "")
        short_desc = doc.metadata.get("short_description")
        page_text = doc.page_content[:300]
        headline = doc.metadata.get("headline", "[No Headline]")

        full_text = extract_article_text(link)

        if full_text:
            content = full_text
        elif short_desc and isinstance(short_desc, str) and len(short_desc.strip()) > 0:
            content = short_desc.strip()
        else:
            content = page_text.strip()

        formatted.append(
            f"Headline: {headline}\n"
            f"Content: {content}\n"
            f"Link: {link}"
        )

    return "\n\n".join(formatted)

# LLM pipeline
chain = (
    RunnableLambda(lambda query: {"query": query, "documents": format_docs(retriever(query))})
    | prompt_template
    | llm
)

def answer_query(query: str):
    docs = retriever(query)[:2]  # Limit to top 2 for performance
    summary = chain.invoke(query)
    return summary.content.strip(), docs
