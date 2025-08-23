"""
RAG API Client for Shadowrun RAG System
Clean API client for document and RAG operations with proper error handling.
"""

import requests
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class RAGClient:
    """Enhanced client for interacting with the FastAPI backend for RAG operations."""

    def __init__(self, api_url: str):
        self.api_url = api_url
        self.active_jobs = {}

    def upload_pdf(self, file_path: str, document_type: str = "rulebook") -> Dict:
        """Upload a PDF with document type specification."""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (Path(file_path).name, f, 'application/pdf')}
                data = {'document_type': document_type}
                response = requests.post(
                    f"{self.api_url}/upload",
                    files=files,
                    data=data,
                    timeout=60
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return {"error": str(e)}

    def get_job_status(self, job_id: str) -> Dict:
        """Poll job status."""
        try:
            response = requests.get(f"{self.api_url}/job/{job_id}", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {"error": str(e)}

    def query_stream(self, question: str, **params) -> tuple:
        """Enhanced stream query response with <think> tag detection."""
        try:
            response = requests.post(
                f"{self.api_url}/query_stream",
                json={"question": question, **params},
                stream=True,
                timeout=30
            )
            response.raise_for_status()

            # ... (rest of the streaming logic from original)
            # This would be copied from your original implementation

        except Exception as e:
            logger.error(f"Query failed: {e}")
            yield f"Error: {str(e)}", "", None, "error"

    def get_models(self) -> List[str]:
        """Get available models."""
        try:
            response = requests.get(f"{self.api_url}/models", timeout=5)
            response.raise_for_status()
            return response.json().get("models", ["llama3"])
        except Exception as e:
            logger.error(f"Failed to fetch models: {e}")
            return ["llama3"]

    def get_documents(self) -> List[str]:
        """Get indexed documents."""
        try:
            response = requests.get(f"{self.api_url}/documents", timeout=10)
            response.raise_for_status()
            return response.json().get("documents", [])
        except Exception as e:
            logger.error(f"Failed to fetch documents: {e}")
            return []

    def get_document_stats(self) -> Dict:
        """Get enhanced document statistics."""
        try:
            response = requests.get(f"{self.api_url}/document_stats", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch document stats: {e}")
            return {"error": str(e)}

    def get_document_content(self, file_path: str) -> Dict:
        """Get content of a specific document."""
        try:
            response = requests.get(f"{self.api_url}/document/{file_path}", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch document content: {e}")
            return {"error": str(e)}

    def search_documents(self, query: str, group: str = None, document_type: str = None,
                         edition: str = None, page: int = 1, page_size: int = 20) -> Dict:
        """Enhanced document search with metadata filtering."""
        try:
            search_params = {
                "query": query,
                "page": page,
                "page_size": page_size
            }
            if group and group != "All Groups":
                search_params["group"] = group
            if document_type and document_type != "All Types":
                search_params["document_type"] = document_type
            if edition and edition != "All Editions":
                search_params["edition"] = edition

            response = requests.post(
                f"{self.api_url}/search_documents",
                json=search_params,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"error": str(e)}

    def get_status(self) -> Dict:
        """Get system status."""
        try:
            response = requests.get(f"{self.api_url}/status", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch status: {e}")
            return {"status": "offline", "error": str(e)}

    def reindex(self, force: bool = False) -> Dict:
        """Trigger reindexing."""
        try:
            response = requests.post(
                f"{self.api_url}/index",
                json={"force_reindex": force},
                timeout=300
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Reindex failed: {e}")
            return {"error": str(e)}