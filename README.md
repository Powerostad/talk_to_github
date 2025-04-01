# GitHub Code RAG

## Overview

GitHub Code RAG is a Retrieval-Augmented Generation (RAG) system that allows users to interactively query a GitHub repository's codebase using natural language. This project utilizes powerful language models for querying, helping users gain insights about codebases easily.

## Features

- Clone and analyze any GitHub repository.
- Process and chunk repository files for efficient querying.
- Create and save vector stores for rapid data retrieval.
- Interactive querying interface with natural language.

## Prerequisites

- Python 3.x
- [dotenv](https://pypi.org/project/python-dotenv/) package
- [FAISS](https://github.com/facebookresearch/faiss) library
- Groq API key (set as `GROQ_API_KEY` in your environment)

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/your-username/GitHub_Code_RAG.git
    cd GitHub_Code_RAG
    ```

2. Install required Python packages:

    ```bash
    pip install -r requirements.txt
    ```
   
3. Set up your environment variables by creating a `.env` file:

    ```
    GROQ_API_KEY=your_groq_api_key_here
    ```

## Usage

1. Run the main script:

    ```bash
    python main.py
    ```

2. When prompted, enter the GitHub repository URL you wish to analyze. If left blank, the default repository is `https://github.com/langchain-ai/langchain`.

3. Follow the interactive prompts to query the repository's codebase:

   - Type any natural language question about the repository.
   - Type 'quit' or 'exit' to end the session.

## Code Structure

- `main.py`: Entry point of the application, handling setup and interactive querying.
- `Github.py`: Contains the `GithubRAG` class, responsible for repository processing and interaction with the vector store.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests for bug fixes, improvements, and new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the teams behind the tools and libraries utilized by this project.
