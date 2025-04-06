import os
import subprocess
import time

def run_once(label, check_file, command):
    """Run a setup command only if its output file/folder does not exist."""
    if not os.path.exists(check_file):
        print(f"🔧 Setting up: {label}")
        subprocess.run(command, shell=True, check=True)
    else:
        print(f"✅ Skipping {label} — already completed.")

def main():
    print("🧠 Bootstrapping News RAG System...")

    # 1. Download spaCy model
    subprocess.run("python -m spacy download en_core_web_sm", shell=True)

    # 2. Data Preprocessing
    run_once("Data Preprocessing", "cleaned_news_final.csv", "python3 data_preprocessing.py")

    # 3. Build FAISS Vector Store
    run_once("FAISS Vector Store", "faiss_index", "python3 build_vector_store.py")

    # 4. Run the main application
    print("🚀 Launching News RAG Application...\n")
    time.sleep(1)
    subprocess.run("python main.py", shell=True)

if __name__ == "__main__":
    main()
