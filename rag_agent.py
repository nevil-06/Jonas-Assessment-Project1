from dotenv import load_dotenv

load_dotenv()

import requests
from bs4 import BeautifulSoup
import tiktoken

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from hybrid_retriever import get_hybrid_retriever

# === Load resources ===
retriever = get_hybrid_retriever()
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.2)
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

# === Prompt Template ===
prompt_template = ChatPromptTemplate.from_template(
    """
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
"""
)

# === Caching for article scraping ===
ARTICLE_CACHE = {}


def extract_article_text(link):
    if link in ARTICLE_CACHE:
        return ARTICLE_CACHE[link]

    try:
        response = requests.get(link, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            article_text = " ".join(p.get_text() for p in paragraphs).strip()
            content = article_text[:1000]  # Max 1000 chars
            ARTICLE_CACHE[link] = content
            return content
    except Exception as e:
        print(f"[Warning] Failed to fetch article from {link}\n{e}")

    ARTICLE_CACHE[link] = None
    return None


# === Token Trimming ===
def trim_to_tokens(text, max_tokens=500):
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        return enc.decode(tokens[:max_tokens])
    return text


# === Document Formatter ===
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
        elif short_desc and isinstance(short_desc, str) and short_desc.strip():
            content = short_desc.strip()
        else:
            content = page_text.strip()

        content = trim_to_tokens(content, max_tokens=500)

        formatted.append(
            f"Headline: {headline}\n" f"Content: {content}\n" f"Link: {link}"
        )

    return "\n\n".join(formatted)


# === RAG Chain ===
chain = (
    RunnableLambda(
        lambda query: {"query": query, "documents": format_docs(retriever(query)[:2])}
    )
    | prompt_template
    | llm
)


def answer_query(query: str):
    docs = retriever(query)[:2]
    summary = chain.invoke(query)
    return summary.content.strip(), docs
