
import streamlit as st
from agent_router import route_request
from ner_utils import extract_named_entities



st.set_page_config(page_title="News RAG System", page_icon="🧠")

st.title("📰 News RAG & LinkedIn Post Generator")

user_input = st.text_area("🔍 Enter a news topic or content", height=200)

# Action buttons
col1, col2, col3 = st.columns(3)
ask_news = col1.button("📚 Ask about News")
gen_post = col2.button("💼 Generate LinkedIn Post")
smart_route = col3.button("🧭 Smart Route")

if user_input:
    if ask_news:
        st.subheader("📚 News Summary")
        result = route_request(user_input, generate_post_only=False, generate_post_from_retrieval=False)

        st.success(result["rag_answer"])

        st.markdown("### 🔗 Source Headlines")
        for doc in result["sources"]:
            st.markdown(f"- **{doc.metadata.get('headline')}**  \n  🔗 {doc.metadata.get('link')}")

        st.markdown("### 🧠 Named Entities")
        for ent, label in extract_named_entities(result["rag_answer"]):
            st.markdown(f"- `{ent}` (**{label}**)")

    elif gen_post:
        st.subheader("💼 LinkedIn Post")
        result = route_request(user_input, generate_post_only=True)
        st.info(result["output"])

    elif smart_route:
        st.subheader("🧠 Summary + Post")
        result = route_request(user_input, generate_post_only=False, generate_post_from_retrieval=True)

        st.success(result["rag_answer"])

        st.markdown("### 🔗 Source Headlines")
        for doc in result["sources"]:
            st.markdown(f"- **{doc.metadata.get('headline')}**  \n  🔗 {doc.metadata.get('link')}")

        st.markdown("### 🧠 Named Entities")
        for ent, label in extract_named_entities(result["rag_answer"]):
            st.markdown(f"- `{ent}` (**{label}**)")

        st.markdown("### 📣 LinkedIn Post")
        st.info(result["linkedin_post"])
