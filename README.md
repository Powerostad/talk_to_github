# GitHub Code RAG System

A simple Retrieval Augmented Generation (RAG) system for analyzing GitHub code repositories using Groq for inference.

## Features

- Clone and process GitHub repositories
- Extract and chunk code files and documentation
- Create vector embeddings using Sentence Transformers
- Store vector embeddings using FAISS
- Query the knowledge base using natural language
- Leverage Groq's powerful LLM models for generation

## Setup

1. Clone this repository
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

## Usage

### Basic Usage

```python
# Initialize the RAG system
from github_rag import GitHubRAG

# Create a new instance with your API key (or use .env)
rag = GitHubRAG()

# Clone a repository
repo_path = rag.clone_github_repo("https://github.com/username/repo", "repos")

# Process the repository into documents
documents = rag.process_repository(repo_path)

# Create document chunks
chunks = rag.create_code_chunks(documents)

# Create the vector store
rag.create_vector_store(chunks)

# Save the vector store
rag.save_vector_store("my_vector_store")

# Query the RAG system
answer = rag.query("How does the authentication system work?")
print(answer)
```

### Loading an Existing Vector Store

```python
# Initialize the RAG system
rag = GitHubRAG()

# Load a previously saved vector store
rag.load_vector_store("my_vector_store")

# Query the RAG system
answer = rag.query("Explain the project structure")
print(answer)
```

## Customization

You can customize the RAG system in several ways:

- Change the Groq model by passing `model_name` to the constructor
- Adjust the chunking strategy in `create_code_chunks()`
- Modify the prompt template in `query()`
- Change the number of retrieved chunks with `k` parameter in `query()`

## Limitations

- Large repositories may take a while to process
- Binary files and very large files are skipped
- The system is designed for code understanding but may not capture all context
- Performance depends on the quality of the embeddings and the LLM