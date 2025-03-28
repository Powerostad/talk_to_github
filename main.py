
import os
from dotenv import load_dotenv
from Github import GithubRAG

def main():
    # Load environment variables
    load_dotenv()

    # Check if API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not set in environment or .env file")
        return

    # Initialize the RAG system
    rag = GithubRAG(groq_api_key=os.environ.get("GROQ_API_KEY"), model_name="deepseek-r1-distill-qwen-32b")  # You can also try mixtral-8x7b-32768

    # Choose a repository to analyze
    repo_url = input("Enter GitHub repository URL (default: https://github.com/langchain-ai/langchain): ")
    repo_url = repo_url.strip() or "https://github.com/langchain-ai/langchain"

    # Determine vector store path
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    vector_store_path = f"{repo_name}_vector_store"

    # Check if vector store already exists
    if os.path.exists(vector_store_path):
        print(f"Loading existing vector store from {vector_store_path}")
        rag.load_vector_store(vector_store_path)
    else:
        print(f"Cloning repository {repo_url}")
        repo_path = rag.clone_github_repository(repo_url, "repos")

        print("Processing repository files")
        documents = rag.process_repository(repo_path)

        print("Creating document chunks")
        chunks = rag.create_code_chunks(documents)

        print("Creating vector store")
        rag.create_vector_store(chunks)

        print(f"Saving vector store to {vector_store_path}")
        rag.save_vector_store(vector_store_path)

    # Interactive query loop
    print("\n=== GitHub Code RAG System ===")
    print("Type 'quit' or 'exit' to end the session")

    while True:
        query = input("\nEnter your question: ")
        if query.lower() in ['quit', 'exit', 'q']:
            break

        if not query.strip():
            continue

        print("\nSearching and generating answer...")
        try:
            answer = rag.query(query)
            print("\nAnswer:")
            print(answer)
        except Exception as e:
            print(f"Error: {e}")

    print("Thank you for using the GitHub Code RAG System!")


if __name__ == "__main__":
    main()