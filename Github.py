import os.path
from typing import Optional, List, Dict, Any

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter


class GithubRAG:
    def __init__(self, groq_api_key: Optional[str] = None, model_name: str = "llama3-70b-8192"):
        """
            Initialize the RAG system with Groq API key and model configuration.

            Args:
                groq_api_key: The API key for Groq (will use from env if not provided)
                model_name: The model name to use from Groq
        """
        self.groq_api_key = groq_api_key
        if self.groq_api_key is None:
            raise ValueError("Groq API key not provided")
        self.model_name = model_name
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            api_key=self.groq_api_key,
            model=self.model_name
        )
        self.vector_store = None

    def clone_github_repository(self, repo_url: str, target_dir: str):
        """
        Clone a GitHub repository to the local filesystem.

        Args:
            repo_url: URL of the GitHub repository
            target_dir: Directory to clone the repository into

        Returns:
            Path to the cloned repository
        """
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        repo_path = os.path.join(target_dir, repo_name)

        if os.path.exists(repo_path):
            print("Repo already exists, skipping clone")
            return repo_path

        import subprocess
        try:
            subprocess.run(["git", "clone", repo_url, repo_path], check=True)
            print("Repo cloned to {}".format(repo_path))
            return repo_path
        except subprocess.CalledProcessError as e:
            print("Failed to clone repository")
            raise e

    def _is_code_file(self, file_path: str) -> bool:
        """
        Determine if a file is a code file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            True if the file is a code file, False otherwise
        """
        code_extensions = [
            '.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp',
            '.cs', '.go', '.rb', '.php', '.swift', '.kt', '.rs',
            '.sh', '.bash', '.html', '.css', '.sql', '.tsx', '.jsx'
        ]
        return any(file_path.endswith(ext) for ext in code_extensions)

    def _is_documentation_file(self, file_path: str) -> bool:
        """
        Determine if a file is a documentation file based on its name or extension.

        Args:
            file_path: Path to the file

        Returns:
            True if the file is a documentation file, False otherwise
        """
        doc_patterns = [
            'readme', 'documentation', 'docs', '.md', '.rst', '.txt',
            'license', 'changelog', 'contributing', 'authors'
        ]
        lower_path = file_path.lower()
        return any(pattern in lower_path for pattern in doc_patterns)

    def process_repository(self, repo_path: str) -> List[Dict[str, Any]]:
        """
        Process a repository to extract code and documentation.

        Args:
            repo_path: Path to the repository

        Returns:
            List of documents with content and metadata
        """
        all_files = []
        for root, _, files in os.walk(repo_path):
            if any(part.startswith(".") for part in root.split(os.sep)) or \
                "node_modules" in root or "venv" in root or "__pycache__" in root:
                continue

            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)

                if os.path.getsize(file_path) > 1_000_000 or file.startswith("."):
                    continue

                if self._is_code_file(file_path) or self._is_documentation_file(file_path):
                    all_files.append((file_path, relative_path))

        documents = []
        for file_path, relative_path in all_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if content.strip(): # Skip empty files
                    doc_type = "code" if self._is_code_file(file_path) else "documentation"
                    documents.append({
                        "content": content,
                        "metadata": {
                            "source": relative_path,
                            "type": doc_type,
                            "extension": os.path.splitext(file_path)[1]
                        }
                    })
            except UnicodeDecodeError:
                # Skip binary files
                continue
        print(f"Processed {len(all_files)} files from repository")
        return documents

    def create_code_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split code documents into appropriate chunks for embedding.

        Args:
            documents: List of documents with content and metadata

        Returns:
            List of document chunks with content and metadata
        """
        code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\nclass ", "\ndef ", "\nfunction ", "\n\n", "\n", " ", ""]
        )
        doc_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
        )

        chunks = []
        for doc in documents:
            content = doc["content"]
            metadata = doc["metadata"]

            splitter = code_splitter if metadata["type"] == "code" else doc_splitter

            doc_chunks = splitter.create_documents(
                texts=[content],
                metadatas=[metadata]
            )

            for chunk in doc_chunks:
                chunks.append({
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                })

        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    def create_vector_store(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Create a vector store from chunks for efficient retrieval.

        Args:
            chunks: List of document chunks with content and metadata
        """
        texts = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embedding_model,
            metadatas=metadatas
        )

        print(f"Created vector store from {len(chunks)} chunks")

    def save_vector_store(self, path: str) -> None:
        """
        Save the vector store to disk.

        Args:
            path: Path to save the vector store to
        """
        if self.vector_store is None:
            raise ValueError("Vector store has not created yet")

        self.vector_store.save_local("vector_dbs/" + path)
        print(f"Saved vector store to {path}")

    def load_vector_store(self, path: str) -> None:
        """
        Load a vector store from disk.

        Args:
            path: Path to load the vector store from
        """
        self.vector_store = FAISS.load_local(
            folder_path="vector_dbs/" + path,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True  # Explicitly allow deserialization
        )
        print(f"Loaded vector store from {path}")

    def query(self, query: str, k: int = 5) -> str:
        """
        Query the RAG system with a natural language query.

        Args:
            query: The question to ask
            k: Number of most relevant chunks to retrieve

        Returns:
            The answer to the query
        """
        if self.vector_store is None:
            raise ValueError("Vector store has not created yet")
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

        # Create prompt template
        prompt_template = """
        You are an expert software engineer tasked with answering questions about a GitHub repository's code.
        Use the following retrieved code snippets and documentation to answer the question.
        If you don't know the answer, say you don't know and explain what information is missing.

        Retrieved context:
        {context}

        Question: {input}

        Your response should be thorough and well-structured. If the answer involves code, include relevant code snippets and explain them.
                """

        # Create prompt
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Create document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)

        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Run the chain
        response = retrieval_chain.invoke({"input": query})

        return response["answer"]

