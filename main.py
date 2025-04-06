# main.py

from agent_router import route_request
from ner_utils import extract_named_entities


def main():
    print("📰 Welcome to the News RAG + LinkedIn Post Generator 🚀")

    while True:
        print("\nChoose an option:")
        print(
            "1. Ask about a news topic (retrieves articles + generates LinkedIn post)"
        )
        print("2. Generate LinkedIn post from your own content")
        print("3. Ask about a news topic (retrieves articles only)")
        print("0. Exit")

        choice = input("Your choice (0/1/2/3): ").strip()

        if choice == "0":
            print("👋 Goodbye!")
            break

        elif choice == "1":
            query = input("Enter your news query: ").strip()
            if not query:
                print("⚠️ Please enter a valid query.")
                continue

            result = route_request(
                query, generate_post_only=False, generate_post_from_retrieval=True
            )

            print("\n🧠 News Summary:\n", result["rag_answer"])
            print("\n🔗 Source Headlines:")
            for doc in result["sources"]:
                headline = doc.metadata.get("headline", "[No Headline]")
                category = doc.metadata.get("category", "N/A")
                link = doc.metadata.get("link", "[No Link]")
                print(f" - {headline} [{category}]\n   🔗 {link}")

            print("\n🧠 Named Entities from Summary:")
            entities = extract_named_entities(result["rag_answer"])
            for text, label in entities:
                print(f" - {text} [{label}]")
            print("\n📣 LinkedIn Post:\n", result["linkedin_post"])

        elif choice == "2":
            custom_content = input("Paste the news content for your LinkedIn post:\n")
            if not custom_content:
                print("⚠️ Please enter some content.")
                continue

            result = route_request(custom_content, generate_post_only=True)
            print("\n📣 LinkedIn Post:\n", result["output"])

        elif choice == "3":
            query = input("Enter your news query: ").strip()
            if not query:
                print("⚠️ Please enter a valid query.")
                continue

            result = route_request(
                query, generate_post_only=False, generate_post_from_retrieval=False
            )

            print("\n🧠 News Summary:\n", result["rag_answer"])
            print("\n🔗 Source Headlines:")
            for doc in result["sources"]:
                headline = doc.metadata.get("headline", "[No Headline]")
                category = doc.metadata.get("category", "N/A")
                link = doc.metadata.get("link", "[No Link]")
                print(f" - {headline} [{category}]\n   🔗 {link}")

        else:
            print("❌ Invalid choice. Please select 0, 1, 2, or 3.")


if __name__ == "__main__":
    main()
