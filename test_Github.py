import subprocess
import pytest
import os
from unittest.mock import MagicMock, patch

from Github import GithubRAG
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TestGithubRAGIntegration:
    def test_clone_github_repository_success(self, tmpdir, monkeypatch):
        repo_url = "https://github.com/example/test_repo.git"
        target_dir = str(tmpdir)
        repo_path = os.path.join(target_dir, 'test_repo')

        # Mock subprocess.run to simulate successful cloning
        def mock_run(args, check=False):
            mock_result = MagicMock()
            mock_result.returncode = 0
            return mock_result
        monkeypatch.setattr(subprocess, "run", mock_run)

        # Create a mock for the GithubRAG class with a fake API key
        with patch('Github.ChatGroq'):  # Mock the ChatGroq dependency
            with patch('Github.HuggingFaceEmbeddings'):  # Mock the HuggingFaceEmbeddings dependency
                rag = GithubRAG(groq_api_key="test_key")
                cloned_path = rag.clone_github_repository(repo_url, target_dir)

        assert cloned_path == repo_path

    def test_process_repository_extracts_code_and_docs(self, tmpdir):
        # Create a dummy repository with code and documentation files
        repo_path = tmpdir.mkdir("test_repo")
        repo_path.join("code.py").write("def hello():\n    print('hello')")
        repo_path.join("README.md").write("# Test Repo")
        repo_path.join("ignore.txt").write("ignore")

        with patch('Github.ChatGroq'):  # Mock the ChatGroq dependency
            with patch('Github.HuggingFaceEmbeddings'):  # Mock the HuggingFaceEmbeddings dependency
                rag = GithubRAG(groq_api_key="test_key")
                documents = rag.process_repository(str(repo_path))

        assert len(documents) == 3
        code_doc = next((doc for doc in documents if doc["metadata"]["source"] == "code.py"), None)
        readme_doc = next((doc for doc in documents if doc["metadata"]["source"] == "README.md"), None)

        ignore_doc = next((doc for doc in documents if doc["metadata"]["source"] == "ignore.txt"), None)

        assert code_doc is not None
        assert readme_doc is not None
        assert ignore_doc is not None
        assert code_doc["content"] == "def hello():\n    print('hello')"
        assert readme_doc["content"] == "# Test Repo"
        assert ignore_doc["content"] == "ignore"

    def test_create_vector_store_success(self, monkeypatch):
        chunks = [
            {"content": "def hello():\n    print('hello')", "metadata": {"source": "code.py"}},
            {"content": "# Test Repo", "metadata": {"source": "README.md"}}
        ]

        # Mock FAISS.from_texts
        mock_faiss = MagicMock()
        monkeypatch.setattr(FAISS, "from_texts", mock_faiss)
        mock_faiss.return_value = MagicMock()  # Return a mock object

        with patch('Github.ChatGroq'):  # Mock the ChatGroq dependency
            with patch('Github.HuggingFaceEmbeddings'):  # Mock the HuggingFaceEmbeddings dependency
                rag = GithubRAG(groq_api_key="test_key")
                rag.embedding_model = MagicMock()  # Mock the embedding model
                rag.create_vector_store(chunks)

        # Verify that FAISS.from_texts was called
        mock_faiss.assert_called_once()

    def test_clone_github_repository_failure(self, tmpdir, monkeypatch):
        repo_url = "https://github.com/example/nonexistent_repo.git"
        target_dir = str(tmpdir)
        repo_path = os.path.join(target_dir, 'nonexistent_repo')

        # Mock subprocess.run to simulate a cloning failure
        def mock_run_failure(args, check=False):
            raise subprocess.CalledProcessError(returncode=1, cmd=["git", "clone", repo_url, repo_path])
        monkeypatch.setattr(subprocess, "run", mock_run_failure)

        with patch('Github.ChatGroq'):  # Mock the ChatGroq dependency
            with patch('Github.HuggingFaceEmbeddings'):  # Mock the HuggingFaceEmbeddings dependency
                rag = GithubRAG(groq_api_key="test_key")

                with pytest.raises(subprocess.CalledProcessError):
                    rag.clone_github_repository(repo_url, target_dir)


    def test_process_repository_skips_large_files(self, tmpdir):
        # Create a dummy repository with a large file
        repo_path = tmpdir.mkdir("test_repo")
        large_file_path = repo_path.join("large_file.txt")
        large_file_path.write("a" * 1_000_001)  # Create a file larger than 1MB
        repo_path.join("small_file.txt").write("small file")

        with patch('Github.ChatGroq'):  # Mock the ChatGroq dependency
            with patch('Github.HuggingFaceEmbeddings'):  # Mock the HuggingFaceEmbeddings dependency
                rag = GithubRAG(groq_api_key="test_key")
                documents = rag.process_repository(str(repo_path))

        assert len(documents) == 1
        assert documents[0]["metadata"]["source"] == "small_file.txt"

    def test_query_before_vector_store_creation(self):
        with patch('Github.ChatGroq'):  # Mock the ChatGroq dependency
            with patch('Github.HuggingFaceEmbeddings'):  # Mock the HuggingFaceEmbeddings dependency
                rag = GithubRAG(groq_api_key="test_key")

                with pytest.raises(ValueError, match="Vector store has not created yet"):
                    rag.query("What is this?")

    def test_create_code_chunks(self):
        documents = [
            {"content": "def hello():\n    print('hello')", "metadata": {"source": "code.py", "type": "code"}},
            {"content": "# Test Repo", "metadata": {"source": "README.md", "type": "documentation"}}
        ]
        with patch('Github.ChatGroq'):  # Mock the ChatGroq dependency
            with patch('Github.HuggingFaceEmbeddings'):  # Mock the HuggingFaceEmbeddings dependency
                rag = GithubRAG(groq_api_key="test_key")
                chunks = rag.create_code_chunks(documents)
        assert len(chunks) >= 2  # Ensure chunks are created
        assert any("hello" in chunk["content"] for chunk in chunks)
        assert any("Test Repo" in chunk["content"] for chunk in chunks)
