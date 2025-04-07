# agent_router.py

from rag_agent import answer_query
from post_generator import generate_linkedin_post


def route_request(
    user_input: str,
    generate_post_only: bool = False,
    generate_post_from_retrieval: bool = True,
):
    """
    Routes user input to the correct agent:
    - If generate_post_only: direct content → Agent 3
    - If not, and generate_post_from_retrieval: query → Agent 2 → Agent 3
    - If not, and no post needed: query → Agent 2 only
    """

    if generate_post_only:
        print("🧭 Routing to Agent 3: LinkedIn Post Generator")
        post = generate_linkedin_post(user_input)
        return {"type": "linkedin_post", "output": post}

    else:
        print("🧭 Routing to Agent 2: News Retrieval")
        answer, sources = answer_query(user_input)

        if generate_post_from_retrieval:
            print("🧭 Routing to Agent 3: LinkedIn Post Generator")
            combined_sources = "\n\n".join(doc.page_content for doc in sources)
            post = generate_linkedin_post(combined_sources)

            return {
                "type": "rag_and_post",
                "rag_answer": answer,
                "linkedin_post": post,
                "sources": sources,
            }
        else:
            return {"type": "rag_only", "rag_answer": answer, "sources": sources}
