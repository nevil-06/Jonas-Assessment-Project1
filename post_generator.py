# post_generator.py

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Prompt template for LinkedIn post
post_prompt = ChatPromptTemplate.from_template("""
You are a professional content writer. Write a LinkedIn post based on the following news content:

"{news_content}"

Make the post:
- Clear and engaging
- Professional in tone
- Suitable for a general LinkedIn audience
- No hashtags or emojis

Return only the post text.
""")

def generate_linkedin_post(news_content: str) -> str:
    prompt = post_prompt.format_messages(news_content=news_content)
    response = llm.invoke(prompt)
    return response.content