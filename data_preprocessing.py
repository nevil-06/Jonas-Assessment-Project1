import pandas as pd
import unicodedata
import os

# File paths
INPUT_FILE = "cleaned_news.csv"
OUTPUT_FILE = "cleaned_news_final.csv"


def normalize_text(text):
    """Remove extra spaces and normalize unicode characters."""
    if pd.isna(text):
        return ""
    return unicodedata.normalize("NFKC", str(text).strip().replace("\n", " "))


def process_row(row):
    """Create a structured text field for embedding."""
    headline = normalize_text(row["headline"])
    category = normalize_text(row["category"])
    short_desc = normalize_text(row["short_description"])

    # Final structure
    return f"Headline: {headline}\nCategory: {category}\nSummary: {short_desc}"


# Check if the output file already exists
if os.path.exists(OUTPUT_FILE):
    print(f"✅ {OUTPUT_FILE} already exists. Skipping preprocessing.")
else:
    print(f"🔄 Processing and saving to {OUTPUT_FILE}...")

    # Load and clean the dataset
    df = pd.read_csv(INPUT_FILE)

    df["headline"] = df["headline"].apply(normalize_text)
    df["category"] = df["category"].apply(normalize_text)
    df["short_description"] = df["short_description"].apply(normalize_text)
    df["link"] = df["link"].apply(normalize_text)

    df["id"] = df.index.map(lambda x: f"id-{x}")
    df["text"] = df.apply(process_row, axis=1)

    # Filter short texts
    df = df[df["text"].str.split().str.len() > 5]

    # Reorder columns
    df = df[["id", "category", "headline", "short_description", "link", "text"]]

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved cleaned and upgraded dataset to {OUTPUT_FILE}")
