# post_generator.py

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.2)

# Prompt template for LinkedIn post
post_prompt = ChatPromptTemplate.from_template(
    """
You are an experienced LinkedIn content strategist.

Your task is to write a **professional, concise, and engaging LinkedIn post** based on the following news content:

------------------
{news_content}
------------------

Instructions:
- Write as if you are a thought leader or industry professional commenting on this news.
- Begin with a strong hook or insight (1–2 sentences).
- Follow with 2–3 sentences that reflect on the impact, relevance, or implications.
- End with a clear takeaway, reflection, or question to engage the reader.
- Keep the tone professional, clear, and insightful.
- DO NOT use hashtags, emojis, or casual/slang language.
- Keep the post under **150 words**.
- Return only the final post content. No explanations, labels, or headings.

Output:
"""
)


def generate_linkedin_post(news_content: str) -> str:
    prompt = post_prompt.format_messages(news_content=news_content)
    response = llm.invoke(prompt)
    return response.content
